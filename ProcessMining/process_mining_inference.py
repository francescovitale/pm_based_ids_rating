from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.obj import EventLog
from pm4py.objects.conversion.log import converter as log_converter

import os
import sys
import warnings
from collections import defaultdict
from statistics import mean
import math
import re
import numpy as np
import json

warnings.filterwarnings("ignore", category=FutureWarning)

input_dir = "Input/PMI/"
input_eventlogs_dir = input_dir + "EventLogs/"
input_petrinets_dir = input_dir + "PetriNets/"
input_metrics_dir = input_dir + "Metrics/"

output_dir = "Output/PMI/"
output_classificationmetrics_dir = output_dir + "ClassificationMetrics/"
output_explanation_dir = output_dir + "Explanation/"

def read_event_logs():
	event_logs = {"TP": {}, "FP": {}}

	for label in ["TP", "FP"]:
		dir_path = os.path.join(input_eventlogs_dir, label)
		if not os.path.exists(dir_path):
			continue
		for file in os.listdir(dir_path):
			if file.endswith(".xes"):
				state = file.split(".xes")[0]
				log_path = os.path.join(dir_path, file)
				event_logs[label][state] = xes_importer.apply(log_path)

	return event_logs

def read_petri_nets():
	petri_nets = {}
	for file in os.listdir(input_petrinets_dir):
		if file.endswith(".pnml"):
			state = file.split(".pnml")[0]
			net, im, fm = pnml_importer.apply(os.path.join(input_petrinets_dir, file))
			petri_nets[state] = {"network": net, "initial_marking": im, "final_marking": fm}
	return petri_nets

def read_fitness_threshold():
	threshold_file = os.path.join(input_metrics_dir, "fitness_threshold.txt")
	with open(threshold_file, "r") as f:
		line = f.readline()
		fitness_threshold = float(line.strip().split(":")[1])
	return fitness_threshold
	
def read_state_similarities_and_aggregates():

	timing_trace_state_similarities = {}
	timing_training_aggregates = {}
	payload_trace_state_similarities = {}
	payload_training_aggregates = {}
	control_flow_trace_state_similarities = {}
	control_flow_training_aggregates = {}

	filepath = os.path.join(input_metrics_dir, "statewise_similarity.txt")
	if not os.path.exists(filepath):
		print(f"Warning: {filepath} not found.")
		return (timing_trace_state_similarities, timing_training_aggregates,
				payload_trace_state_similarities, payload_training_aggregates,
				control_flow_trace_state_similarities, control_flow_training_aggregates)

	current_state = None
	current_section = None 

	with open(filepath, "r") as f:
		for line in f:
			line_clean = line.strip()
			if not line_clean:
				current_section = None
				continue

			state_match = re.match(r"^===\s*(STATE_\d+)\s*===$", line_clean)
			if state_match:
				current_state = state_match.group(1)
				current_section = None
				continue

			if current_state is None:
				continue

			if line_clean.startswith("Control-flow similarity:"):
				current_section = "control_similarity"
				continue
			if line_clean.startswith("Control-flow training aggregates:"):
				current_section = "control_aggregates"
				control_flow_training_aggregates.setdefault(current_state, {})
				continue

			if line_clean.startswith("Timing similarity:"):
				current_section = "timing_similarity"
				continue
			if line_clean.startswith("Timing training aggregates"):
				current_section = "timing_aggregates"
				timing_training_aggregates.setdefault(current_state, {})
				continue

			if line_clean.startswith("Payload similarity:"):
				current_section = "payload_similarity"
				continue
			if line_clean.startswith("Payload training aggregates"):
				current_section = "payload_aggregates"
				payload_training_aggregates.setdefault(current_state, {})
				continue

			if current_section in ["control_similarity", "timing_similarity", "payload_similarity"]:
				try:
					sim_val = float(line_clean)
				except ValueError:
					print(f"Warning: could not parse similarity for {current_state}: '{line_clean}'")
					continue

				if current_section == "control_similarity":
					control_flow_trace_state_similarities[current_state] = sim_val
				elif current_section == "timing_similarity":
					timing_trace_state_similarities[current_state] = sim_val
				elif current_section == "payload_similarity":
					payload_trace_state_similarities[current_state] = sim_val

			elif current_section in ["control_aggregates", "timing_aggregates", "payload_aggregates"]:
				if ":" not in line_clean:
					continue
				activity, val = line_clean.split(":", 1)
				activity = activity.strip()
				val = val.strip()
				try:
					val_float = float(val)
				except ValueError:
					print(f"Warning: could not parse aggregate for {current_state}/{activity}: '{val}'")
					continue

				if current_section == "control_aggregates":
					control_flow_training_aggregates[current_state][activity] = val_float
				elif current_section == "timing_aggregates":
					timing_training_aggregates[current_state][activity] = val_float
				elif current_section == "payload_aggregates":
					payload_training_aggregates[current_state][activity] = val_float

	return (timing_trace_state_similarities, timing_training_aggregates,
			payload_trace_state_similarities, payload_training_aggregates,
			control_flow_trace_state_similarities, control_flow_training_aggregates)
	
def compute_event_alignments(petri_nets, event_logs):
	trace_state_alignments = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

	for label, state_logs in event_logs.items():
		for state, log in state_logs.items():

			if not log or state not in petri_nets:
				continue

			try:
				_, aligned_traces = compute_fitness(
					petri_nets[state],
					log,
					cc_variant="ALIGNMENT_BASED"
				)
			except Exception:
				_, aligned_traces = compute_fitness(
					petri_nets[state],
					log,
					cc_variant="TOKEN_BASED"
				)

			for i, alignment in enumerate(aligned_traces):
				trace = log[i]
				trace_id = trace.attributes.get("concept:name")
				if trace_id is None:
					continue

				aligned_moves = list(alignment.values())[0]

				for log_move, model_move in aligned_moves:
					if log_move == model_move and log_move not in [None, ">>"]:
						trace_state_alignments[label][trace_id][state][log_move] += 1

	return trace_state_alignments

def compute_fitness(petri_net, event_log, cc_variant):

	log_fitness = 0.0
	aligned_traces = None
	parameters = {}
	parameters[log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY] = 'CaseID'
	
	if cc_variant == "ALIGNMENT_BASED":
		aligned_traces = alignments.apply_log(event_log, petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
		log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
	elif cc_variant == "TOKEN_BASED":
		replay_results = tokenreplay.algorithm.apply(log = event_log, net = petri_net["network"], initial_marking = petri_net["initial_marking"], final_marking = petri_net["final_marking"], parameters = parameters, variant = tokenreplay.algorithm.Variants.TOKEN_REPLAY)
		log_fitness = replay_fitness.evaluate(results = replay_results, variant = replay_fitness.Variants.TOKEN_BASED)["log_fitness"]

	return log_fitness, aligned_traces	

def compute_trace_fitness(event_logs, petri_nets):
	parameters = {
		log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "CaseID"
	}

	trace_fitness_map = defaultdict(list)

	trace_label_map = {}

	for label, logs in event_logs.items():
		for state, log in logs.items():
			if log is None or state not in petri_nets:
				continue

			net_info = petri_nets[state]

			for trace in log:
				trace_id = trace.attributes.get("concept:name")
				if trace_id is None:
					continue

				trace_label_map[trace_id] = label

				single_trace_log = EventLog([trace])

				try:
					aligned_traces = alignments.apply_log(
						single_trace_log,
						net_info["network"],
						net_info["initial_marking"],
						net_info["final_marking"],
						parameters=parameters,
						variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR
					)
					fitness = replay_fitness.evaluate(
						aligned_traces,
						variant=replay_fitness.Variants.ALIGNMENT_BASED
					)["log_fitness"]
				except Exception:
					fitness = 0.0

				trace_fitness_map[trace_id].append(fitness)

	tp_fitness_values = []
	fp_fitness_values = []

	for trace_id, fitness_list in trace_fitness_map.items():
		min_fitness = min(fitness_list)
		if trace_label_map.get(trace_id) == "TP":
			tp_fitness_values.append(min_fitness)
		elif trace_label_map.get(trace_id) == "FP":
			fp_fitness_values.append(min_fitness)

	return tp_fitness_values, fp_fitness_values	

def extract_event_time_distributions(event_logs):

	trace_state_timings = defaultdict(
		lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
	)

	for label, state_logs in event_logs.items():
		for state, log in state_logs.items():
			if log is None:
				continue

			for trace in log:
				trace_id = trace.attributes.get("concept:name")
				if trace_id is None:
					continue

				events = list(trace)

				for i in range(len(events) - 1):
					curr = events[i]
					nxt = events[i + 1]

					activity = curr["concept:name"]
					t1 = curr.get("time:timestamp")
					t2 = nxt.get("time:timestamp")

					if t1 is None or t2 is None:
						continue

					duration = (t2 - t1).total_seconds()
					if duration < 0:
						continue

					trace_state_timings[label][trace_id][state][activity].append(duration)

	return trace_state_timings
	
def extract_trace_state_payloads(event_logs):
	trace_state_payloads = defaultdict(
		lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
	)

	for label, state_logs in event_logs.items():
		for state, log in state_logs.items():
			if not log:
				continue

			for trace in log:
				trace_id = trace.attributes.get("concept:name")
				if trace_id is None:
					continue

				for event in trace:
					activity = event["concept:name"]
					payload = event.get("payload_size", 0.0)
					try:
						payload = float(payload)
					except (ValueError, TypeError):
						payload = 0.0

					trace_state_payloads[label][trace_id][state][activity].append(payload)

	return trace_state_payloads	

def compute_trace_timing_similarities(trace_state_timings,timing_training_aggregates):


	def compute_similarity(trace_timings, train_agg):
		all_activities = set(train_agg.keys()) | set(trace_timings.keys())

		vec_train = [train_agg.get(act, 0.0) for act in all_activities]
		vec_trace = [trace_timings.get(act, 0.0) for act in all_activities]

		dot_product = sum(a * b for a, b in zip(vec_train, vec_trace))
		norm_train = math.sqrt(sum(a ** 2 for a in vec_train))
		norm_trace = math.sqrt(sum(b ** 2 for b in vec_trace))

		return dot_product / (norm_train * norm_trace) if norm_train > 0 and norm_trace > 0 else 0.0

	tp_similarities = {}
	fp_similarities = {}

	for label, traces in trace_state_timings.items():  # TP / FP
		for trace_id, states in traces.items():
			for state, trace_timings in states.items():

				train_agg = timing_training_aggregates.get(state)
				if not train_agg:
					continue

				similarity = compute_similarity(trace_timings, train_agg)

				if label == "TP":
					tp_similarities.setdefault(trace_id, {})[state] = similarity
				elif label == "FP":
					fp_similarities.setdefault(trace_id, {})[state] = similarity

	return tp_similarities, fp_similarities

def aggregate_trace_timings(trace_state_timings):

	aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

	for label, traces in trace_state_timings.items():
		for trace_id, states in traces.items():
			for state, activities in states.items():
				for activity, durations in activities.items():
					if durations:
						aggregated[label][trace_id][state][activity] = sum(durations) / len(durations)
					else:
						aggregated[label][trace_id][state][activity] = 0.0

	return aggregated
	
def aggregate_trace_state_payloads(trace_state_payloads):
	aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

	for label, traces in trace_state_payloads.items():
		for trace_id, states in traces.items():
			for state, activities in states.items():
				for activity, payload_list in activities.items():
					if payload_list:
						aggregated[label][trace_id][state][activity] = sum(payload_list) / len(payload_list)
					else:
						aggregated[label][trace_id][state][activity] = 0.0

	return aggregated
	
def compute_trace_payload_similarities(trace_state_payloads, payload_training_aggregates):

	def compute_similarity(trace_payloads, train_agg):
		all_activities = set(train_agg.keys()) | set(trace_payloads.keys())

		vec_train = [train_agg.get(act, 0.0) for act in all_activities]
		vec_trace = [trace_payloads.get(act, 0.0) for act in all_activities]

		dot_product = sum(a * b for a, b in zip(vec_train, vec_trace))
		norm_train = math.sqrt(sum(a ** 2 for a in vec_train))
		norm_trace = math.sqrt(sum(b ** 2 for b in vec_trace))

		return dot_product / (norm_train * norm_trace) if norm_train > 0 and norm_trace > 0 else 0.0

	tp_similarities = {}
	fp_similarities = {}

	for label, traces in trace_state_payloads.items():
		for trace_id, states in traces.items():
			for state, trace_payloads in states.items():

				train_agg = payload_training_aggregates.get(state)
				if not train_agg:
					continue

				similarity = compute_similarity(trace_payloads, train_agg)

				if label == "TP":
					tp_similarities.setdefault(trace_id, {})[state] = similarity
				elif label == "FP":
					fp_similarities.setdefault(trace_id, {})[state] = similarity

	return tp_similarities, fp_similarities

def analyze_traces_with_alarm_bands(trace_state_alignments,training_aggregates,state_thresholds):

	def compute_similarity(trace_activities, train_agg):
		all_activities = set(train_agg.keys()) | set(trace_activities.keys())
		vec_train = [train_agg.get(act, 0.0) for act in all_activities]
		vec_trace = [trace_activities.get(act, 0.0) for act in all_activities]

		dot_product = sum(a * b for a, b in zip(vec_train, vec_trace))
		norm_train = math.sqrt(sum(a ** 2 for a in vec_train))
		norm_trace = math.sqrt(sum(b ** 2 for b in vec_trace))

		return dot_product / (norm_train * norm_trace) if norm_train > 0 and norm_trace > 0 else 0.0

	trace_similarities = {}
	trace_activity_records = {}
	all_tp_trace_ids = set()
	all_fp_trace_ids = set()

	for label, traces in trace_state_alignments.items():
		for trace_id, states in traces.items():
			if label == "TP":
				all_tp_trace_ids.add(trace_id)
			elif label == "FP":
				all_fp_trace_ids.add(trace_id)

			per_state_sims = {}
			for state, trace_activities in states.items():
				train_agg = training_aggregates.get(state)
				if train_agg is None:
					raise KeyError(f"State '{state}' not found in training_aggregates")

				sim = compute_similarity(trace_activities, train_agg)
				per_state_sims[state] = sim

				trace_activity_records.setdefault(trace_id, {"label": label, "activities": defaultdict(list)})
				for act, cnt in trace_activities.items():
					trace_activity_records[trace_id]["activities"][act].append(cnt)

			trace_similarities[trace_id] = {"label": label, "state_sims": per_state_sims}

	if not trace_similarities:
		return {}

	for record in trace_similarities.values():
		mean_sim = np.mean(list(record["state_sims"].values()))
		record["mean_similarity"] = mean_sim

	def assign_band_absolute(mean_sim):
		if mean_sim <= 0.01:
			return "BAND_0_1"
		elif mean_sim <= 0.25:
			return "BAND_1_25"
		elif mean_sim <= 0.75:
			return "BAND_25_75"
		elif mean_sim <= 0.99:
			return "BAND_75_99"
		else:  # > 0.99
			return "BAND_99_100"

	for trace_id, record in trace_similarities.items():
		record["band"] = assign_band_absolute(record["mean_similarity"])

	bands = [
		"BAND_0_1",
		"BAND_1_25",
		"BAND_25_75",
		"BAND_75_99",
		"BAND_99_100"
	]
	band_stats = {b: {
		"tp_count": 0,
		"fp_count": 0,
		"recall": 0.0,
		"precision": 0.0,
		"total_alignments": {"TP": [], "FP": []},
		"activities": {"TP": defaultdict(list), "FP": defaultdict(list)},
		"similarities": {"TP": [], "FP": []}
	} for b in bands}

	tp_similarities = {}
	fp_similarities = {}

	for trace_id, record in trace_similarities.items():
		label = record["label"]
		band = record["band"]
		state_sims = record["state_sims"]
		mean_sim = record["mean_similarity"]

		total_align = np.mean([np.mean(cnts) for cnts in trace_activity_records[trace_id]["activities"].values()])

		band_stats[band]["total_alignments"][label].append(total_align)
		for act, counts in trace_activity_records[trace_id]["activities"].items():
			band_stats[band]["activities"][label][act].extend(counts)
		band_stats[band]["similarities"][label].append(mean_sim)

		if label == "TP":
			band_stats[band]["tp_count"] += 1
			tp_similarities[trace_id] = mean_sim
		elif label == "FP":
			band_stats[band]["fp_count"] += 1
			fp_similarities[trace_id] = mean_sim

	results = {}

	tp_per_band = [band_stats[b]["tp_count"] for b in bands]
	fp_per_band = [band_stats[b]["fp_count"] for b in bands]

	N = len(bands)

	for k in range(1, N + 1):
		tp_kept = sum(tp_per_band[:k])
		tp_suppressed = sum(tp_per_band[k:])

		recall_denom = tp_kept + tp_suppressed
		recall_sigma_k = tp_kept / recall_denom if recall_denom > 0 else 0.0

		fp_kept = sum(fp_per_band[:k])

		precision_denom = tp_kept + fp_kept
		precision_sigma_k = tp_kept / precision_denom if precision_denom > 0 else 0.0

		tp_bands = {bands[i]: tp_per_band[i] for i in range(N)}
		fp_bands = {bands[i]: fp_per_band[i] for i in range(N)}

		band_stats_dict = {}

		for band in bands:
			stats = band_stats[band]

			band_stats_dict[band] = {
				"mean_total_alignments": {
					label: float(np.mean(stats["total_alignments"][label]))
					if stats["total_alignments"][label] else 0.0
					for label in ["TP", "FP"]
				},
				"mean_alignments_per_activity": {
					label: {
						act: float(np.mean(cnts))
						for act, cnts in stats["activities"][label].items()
					}
					for label in ["TP", "FP"]
				},
				"mean_similarity": {
					label: float(np.mean(stats["similarities"][label]))
					if stats["similarities"][label] else 0.0
					for label in ["TP", "FP"]
				}
			}

		results[k] = {
			"band_threshold": bands[k-1],
			"recall_sigma": recall_sigma_k,
			"precision_sigma": precision_sigma_k,
			"tp_bands": tp_bands,
			"fp_bands": fp_bands,
			"band_stats": band_stats_dict,
		}

	return results, tp_similarities, fp_similarities

def save_analysis_results(results, tp_similarities, fp_similarities, prefix="analysis"):

    results_file = os.path.join(output_classificationmetrics_dir, f"{prefix}_results.txt")
    tp_file = os.path.join(output_classificationmetrics_dir, f"{prefix}_tp_similarities.txt")
    fp_file = os.path.join(output_classificationmetrics_dir, f"{prefix}_fp_similarities.txt")

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    with open(tp_file, "w") as f:
        json.dump(tp_similarities, f, indent=4)
    
    with open(fp_file, "w") as f:
        json.dump(fp_similarities, f, indent=4)

    print(f"Saved results to:\n{results_file}\n{tp_file}\n{fp_file}")


event_logs = read_event_logs()
petri_nets = read_petri_nets()

timing_trace_state_similarities, timing_training_aggregates, payload_trace_state_similarities, payload_training_aggregates, control_flow_trace_state_similarities, control_flow_training_aggregates = read_state_similarities_and_aggregates()
trace_state_alignments = compute_event_alignments(petri_nets, event_logs)
#results = compute_trace_similarities(trace_state_alignments, control_flow_training_aggregates, control_flow_trace_state_similarities)
#precision, recall, TP_count, FP_count, FN_count = compute_precision_recall_control_flow(tp_similarity_values_alignments, fp_similarity_values_alignments, control_flow_trace_state_similarities)
results, tp_similarities, fp_similarities = analyze_traces_with_alarm_bands(trace_state_alignments,control_flow_training_aggregates,control_flow_trace_state_similarities)
save_analysis_results(results, tp_similarities, fp_similarities)


