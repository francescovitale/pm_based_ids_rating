
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import sys
import math
import socket
import pickle 
import pandas as pd
import numpy as np

from collections import defaultdict
from datetime import datetime, timezone

import dpkt
import pm4py

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler


BASE_INPUT_DIR = "Input/ELE"
BASE_OUTPUT_DIR = "Output/ELE"
MODEL_DIR = BASE_OUTPUT_DIR + "/Models"


TCP_FLAGS = {
    dpkt.tcp.TH_SYN: "SYN",
    dpkt.tcp.TH_ACK: "ACK",
    dpkt.tcp.TH_FIN: "FIN",
    dpkt.tcp.TH_RST: "RST",
    dpkt.tcp.TH_PUSH: "PSH",
    dpkt.tcp.TH_URG: "URG"
}

FEATURES = [
    "AVG_PAYLOAD", "N_SERVERS", "N_USER_PORTS",
    "N_ACK", "N_SYN", "N_FIN", "N_PSH", "N_RST"
]

def read_data(input_dir):
    flow_data = {}

    if not os.path.exists(input_dir):
        return flow_data

    for pcap_file in os.listdir(input_dir):
        if not pcap_file.endswith(".pcap"):
            continue

        records = []
        client_ip = None

        try:
            with open(os.path.join(input_dir, pcap_file), "rb") as f:
                pcap = dpkt.pcap.Reader(f)

                for ts, buf in pcap:
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        ip = eth.data
                        tcp = ip.data
                    except Exception:
                        continue

                    if not isinstance(ip, dpkt.ip.IP) or not isinstance(tcp, dpkt.tcp.TCP):
                        continue

                    src_ip = socket.inet_ntoa(ip.src)
                    dst_ip = socket.inet_ntoa(ip.dst)

                    if client_ip is None:
                        client_ip = src_ip if tcp.sport > tcp.dport else dst_ip

                    direction = "C_to_S" if src_ip == client_ip else "S_to_C"

                    flags = [name for bit, name in TCP_FLAGS.items() if tcp.flags & bit]
                    tcp_flag = "+".join(flags) if flags else "OTHER"

                    records.append({
                        "TIMESTAMP": ts,
                        "DIRECTION": direction,
                        "SOURCE_IP": src_ip,
                        "DESTINATION_IP": dst_ip,
                        "SOURCE_PORT": tcp.sport,
                        "DESTINATION_PORT": tcp.dport,
                        "TCP_FLAG": tcp_flag,
                        "PAYLOAD_SIZE": len(tcp.data)
                    })

            df = pd.DataFrame(records)
            if not df.empty:
                flow_data[pcap_file] = df

        except Exception as e:
            print(f"[ERROR] Reading {pcap_file}: {e}")

    return flow_data

def feature_extraction(flow_data, window_size):
    enriched = {}

    for filename, df in flow_data.items():
        if len(df) < window_size:
            continue

        client_ip = df.iloc[0]["SOURCE_IP"]
        rows = []

        n_windows = len(df) // window_size

        for w in range(n_windows):
            window = df.iloc[w * window_size:(w + 1) * window_size]

            avg_payload = window["PAYLOAD_SIZE"].mean()

            ips = set(window["SOURCE_IP"]) | set(window["DESTINATION_IP"])
            ips.discard(client_ip)

            ports = set(window["SOURCE_PORT"]) | set(window["DESTINATION_PORT"])
            ports.discard(443)

            counts = {k: 0 for k in ["ACK", "SYN", "FIN", "PSH", "RST"]}
            for flag in window["TCP_FLAG"]:
                for k in counts:
                    if k in flag:
                        counts[k] += 1

            feature_row = [
                avg_payload, len(ips), len(ports),
                counts["ACK"], counts["SYN"],
                counts["FIN"], counts["PSH"], counts["RST"]
            ]

            rows.extend([feature_row] * window_size)

        fe_df = pd.DataFrame(rows, columns=FEATURES)
        df = df.iloc[:len(fe_df)].reset_index(drop=True)
        enriched[filename] = pd.concat([df, fe_df], axis=1)

    return enriched

def get_scaler(method):
    if method == "zscore":
        return StandardScaler()
    elif method == "min-max":
        return MinMaxScaler()
    raise ValueError("Unknown normalization method")

def get_clusterer(ctype, n_clusters):
    if ctype == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=0)
    if ctype == "gmm":
        return GaussianMixture(n_components=n_clusters, random_state=0)
    if ctype == "agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters)
    raise ValueError("Unknown clustering type")

def save_models(scaler, model, run_id="latest"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, f"scaler_{run_id}.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, f"model_{run_id}.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Models saved to {MODEL_DIR}")

def load_models(run_id="latest"):
    s_path = os.path.join(MODEL_DIR, f"scaler_{run_id}.pkl")
    m_path = os.path.join(MODEL_DIR, f"model_{run_id}.pkl")
    
    if not os.path.exists(s_path) or not os.path.exists(m_path):
        raise FileNotFoundError("Saved models not found. Run 'training' mode first.")
        
    with open(s_path, "rb") as f:
        scaler = pickle.load(f)
    with open(m_path, "rb") as f:
        model = pickle.load(f)
    return scaler, model


def process_features(flow_data, mode, norm_method, clust_type, n_clusters):
    all_features = []
    order = []

    for fname, df in flow_data.items():
        all_features.append(df[FEATURES])
        order.append((fname, len(df)))

    if not all_features:
        return flow_data

    merged = pd.concat(all_features, ignore_index=True)
    
    if mode == "training":
        scaler = get_scaler(norm_method)
        model = get_clusterer(clust_type, n_clusters)
        
        normalized_data = scaler.fit_transform(merged)
        
        if clust_type == "agglomerative":
            labels = model.fit_predict(normalized_data)
        else:
            model.fit(normalized_data)
            labels = model.predict(normalized_data)
            
        save_models(scaler, model)
        
    elif mode == "inference":
        if clust_type == "agglomerative":
            raise ValueError("Agglomerative Clustering does not support inference (prediction) on new data.")
            
        scaler, model = load_models()
        
        normalized_data = scaler.transform(merged)
        labels = model.predict(normalized_data)
        
    else:
        raise ValueError("Invalid mode. Use 'training' or 'inference'")

    idx = 0
    for fname, size in order:
        flow_data[fname]["Cluster"] = labels[idx:idx + size]
        idx += size

    return flow_data


def build_timestamp(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc)

def apply_variant_filtering(log, threshold):
    if log is None or len(log) == 0:
        return log
    
    if abs(threshold - 1.0) < 0.001:
        return log

    try:
        from pm4py.algo.filtering.log.variants import variants_filter
        
        if threshold < 1.0:
            print(f"   > Filtering variants by coverage: {threshold * 100:.1f}%")
            return variants_filter.filter_log_variants_percentage(log, percentage=threshold)
            
        elif threshold > 1.0:
            k = int(threshold)
            print(f"   > Filtering Top-{k} variants")
            return variants_filter.filter_log_variants_top_k(log, k)

    except (ImportError, AttributeError):
        pass 

    try:
        print(f"   > [Fallback] Manually filtering variants...")
        from collections import defaultdict
        
        variants = defaultdict(list)
        for trace in log:
            variant_key = tuple(str(e["concept:name"]) for e in trace)
            variants[variant_key].append(trace)

        sorted_variants = sorted(variants.values(), key=len, reverse=True)
        kept_traces = []

        if threshold > 1.0:
            k = int(threshold)
            print(f"   > Keeping Top-{k} variants (Manual)")
            for group in sorted_variants[:k]:
                kept_traces.extend(group)
                
        elif threshold < 1.0:
            target_count = len(log) * threshold
            current_count = 0
            print(f"   > Keeping {threshold*100:.1f}% coverage (Manual)")
            
            for group in sorted_variants:
                kept_traces.extend(group)
                current_count += len(group)
                if current_count >= target_count:
                    break
        
        return type(log)(kept_traces)

    except Exception as e:
        print(f"[WARN] Manual filtering failed: {e}. Returning original log.")
        return log

def extract_state_event_logs(flow_data, n_clusters, window_size, filter_threshold=1.0):
    state_logs = {}

    for i in range(n_clusters):
        events = []

        for fname, df in flow_data.items():
            if "Cluster" not in df.columns:
                continue

            for w in range(len(df) // window_size):
                window = df.iloc[w * window_size:(w + 1) * window_size]

                if int(window["Cluster"].iloc[0]) != i:
                    continue

                for _, r in window.iterrows():
                    events.append({
                        "case:concept:name": fname,
                        "concept:name": f"{r.DIRECTION}_{r.TCP_FLAG}",
                        "time:timestamp": build_timestamp(r.TIMESTAMP),

                        "payload_size": int(r.PAYLOAD_SIZE)
                    })

        if events:
            event_df = pd.DataFrame(events)

            event_df = pm4py.format_dataframe(
                event_df,
                case_id="case:concept:name",
                activity_key="concept:name",
                timestamp_key="time:timestamp"
            )

            log = pm4py.convert_to_event_log(event_df)
            log = apply_variant_filtering(log, filter_threshold)

            state_logs[f"STATE_{i}"] = log
        else:
            state_logs[f"STATE_{i}"] = None

    return state_logs

def save_event_logs(event_logs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for state, log in event_logs.items():
        if log and len(log) > 0:
            pm4py.write_xes(log, os.path.join(output_dir, f"{state}.xes"))

def run_pipeline(mode, window, normalization, clustering, n_clusters, filter_threshold):
    if mode == "training":
        target_labels = ["FP"]
    else:
        target_labels = ["FP", "TP"]
    
    print(f"\n=== Starting Pipeline in [{mode.upper()}] mode for: {target_labels} ===")

    
    for label in target_labels:
        print(f"\n--- Processing Label: {label} ---")
        input_dir = os.path.join(BASE_INPUT_DIR, label)
        output_dir = os.path.join(BASE_OUTPUT_DIR, label)

        flows = read_data(input_dir)
        if not flows:
            print(f"[INFO] {label}: no data found. Skipping.")
            continue

        flows = feature_extraction(flows, window)
        if not flows:
            print(f"[INFO] {label}: insufficient features. Skipping.")
            continue

        try:
            flows = process_features(flows, mode, normalization, clustering, n_clusters)
        except Exception as e:
            print(f"[ERROR] processing features: {e}")
            continue

        state_logs = extract_state_event_logs(flows, n_clusters, window, filter_threshold)
        
        if not any(log for log in state_logs.values()):
            print(f"[INFO] {label}: no event logs generated.")
            continue

        save_event_logs(state_logs, output_dir)
        print(f"[OK] {label}: logs saved.")


if __name__ == "__main__":
    try:
        MODE = sys.argv[1].lower() 
        FE_WINDOW = int(sys.argv[2])
        NORMALIZATION = sys.argv[3]
        CLUSTERING = sys.argv[4]
        N_CLUSTERS = int(sys.argv[5])
        VARIANT_THRESHOLD = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
        
        if MODE not in ["training", "inference"]:
            raise ValueError("Mode must be 'training' or 'inference'")

    except Exception as e:
        print(f"Error parsing args: {e}")
        print("Usage: python script.py <training|inference> <window> <zscore|min-max> <kmeans|gmm> <n_clusters> [threshold]")
        sys.exit(1)

    run_pipeline(MODE, FE_WINDOW, NORMALIZATION, CLUSTERING, N_CLUSTERS, VARIANT_THRESHOLD)