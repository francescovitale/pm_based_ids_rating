from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.log.obj import EventLog
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pm4py
import random
import os
import sys
import math
import random
import pandas as pd
import numpy as np
from itertools import permutations
from copy import deepcopy
from statistics import mean
from math import sqrt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import time
import func_timeout

from collections import defaultdict


input_dir = "Input/PMT/"
input_eventlogs_dir = input_dir + "EventLogs/"

output_dir = "Output/PMT/"
output_petrinets_dir = output_dir + "PetriNets/"
output_metrics_dir = output_dir + "Metrics/"

variant = ""

def read_event_logs():

	event_logs = {}

	for state in os.listdir(input_eventlogs_dir):
		event_logs[state.split(".xes")[0]] = xes_importer.apply(input_eventlogs_dir + "/" + state)

	return event_logs
	
def split_event_logs(event_logs, validation_percentage):

	training_event_logs = {}
	validation_event_logs = {}

	random.seed(42)

	for state, log in event_logs.items():
		if log is None:
			training_event_logs[state] = None
			validation_event_logs[state] = None
			continue

		traces = list(log)
		random.shuffle(traces)

		n_total = len(traces)
		n_val = int(n_total * validation_percentage)

		train_traces = traces[:n_total - n_val]
		val_traces = traces[n_total - n_val:]

		training_event_logs[state] = EventLog(train_traces)
		validation_event_logs[state] = EventLog(val_traces)

	return training_event_logs, validation_event_logs
	
def extract_event_time_distributions(event_logs):
    trace_state_timings = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for state, log in event_logs.items():
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
                t1 = curr["time:timestamp"]
                t2 = nxt["time:timestamp"]

                if t1 is None or t2 is None:
                    continue

                duration = (t2 - t1).total_seconds()
                if duration < 0:
                    continue

                trace_state_timings[trace_id][state][activity].append(duration)

    return trace_state_timings
	
def cosine_similarity_per_state(training_event_alignments, validation_event_alignments):

    state_similarities = {}
    training_aggregates = {}

    for state in validation_event_alignments:
        if state not in training_event_alignments:
            continue

        training_traces = training_event_alignments[state]
        training_agg = aggregate_event_alignments(training_traces)
        training_aggregates[state] = training_agg

        validation_traces = validation_event_alignments[state]

        similarities = []

        for trace_id, val_dist in validation_traces.items():
            all_events = set(training_agg.keys()) | set(val_dist.keys())
            vec_train = [training_agg.get(e, 0) for e in all_events]
            vec_val = [val_dist.get(e, 0) for e in all_events]

            dot_product = sum(a * b for a, b in zip(vec_train, vec_val))
            norm_train = (sum(a ** 2 for a in vec_train)) ** 0.5
            norm_val = (sum(b ** 2 for b in vec_val)) ** 0.5

            sim = dot_product / (norm_train * norm_val) if norm_train > 0 and norm_val > 0 else 0.0
            similarities.append(sim)

        state_similarities[state] = sum(similarities) / len(similarities) if similarities else 0.0

    return state_similarities, training_aggregates
	
def compute_event_alignments(petri_nets, event_logs):
    trace_activity_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for state, log in event_logs.items():
        if not log or state not in petri_nets:
            continue

        try:
            _, aligned_traces = compute_fitness(
                petri_nets[state], log, cc_variant="ALIGNMENT_BASED"
            )
        except Exception:
            _, aligned_traces = compute_fitness(
                petri_nets[state], log, cc_variant="TOKEN_BASED"
            )

        for i, alignment in enumerate(aligned_traces):
            trace = log[i]
            trace_id = trace.attributes.get("concept:name")
            if trace_id is None:
                continue

            aligned_moves = list(alignment.values())[0]

            for log_move, model_move in aligned_moves:
                if log_move == model_move and log_move not in [None, ">>"]:
                    trace_activity_counts[state][trace_id][log_move] += 1

    return trace_activity_counts
	
def aggregate_event_alignments(trace_activity_counts):

    activity_sum = defaultdict(float)
    trace_count = defaultdict(int)

    for trace_id, activities in trace_activity_counts.items():
        for activity, count in activities.items():
            activity_sum[activity] += count
            trace_count[activity] += 1

    return {
        activity: int(activity_sum[activity] / trace_count[activity])
        for activity in activity_sum
    }		
	
	
def process_discovery(event_logs, variant, noise_threshold):

	petri_nets = {}
	
	for state in event_logs:
		petri_nets[state] = {}
		if variant == "im":
			petri_nets[state]["network"], petri_nets[state]["initial_marking"], petri_nets[state]["final_marking"] = pm4py.discover_petri_net_inductive(event_logs[state], noise_threshold = noise_threshold)
		elif variant == "ilp":
			petri_nets[state]["network"], petri_nets[state]["initial_marking"], petri_nets[state]["final_marking"] = pm4py.discover_petri_net_ilp(event_logs[state], alpha=1-noise_threshold)	
		
	return petri_nets	



	
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

def cosine_similarity_per_state_timing(training_timings, validation_timings):
    training_aggregates = defaultdict(dict)
    for trace_id, states in training_timings.items():
        for state, activities in states.items():
            for activity, durations in activities.items():
                training_aggregates[state][activity] = training_aggregates[state].get(activity, 0.0) + sum(durations)
    
    validation_aggregates = defaultdict(dict)
    for trace_id, states in validation_timings.items():
        for state, activities in states.items():
            for activity, durations in activities.items():
                validation_aggregates[state][activity] = validation_aggregates[state].get(activity, 0.0) + sum(durations)
    
    state_similarities = {}
    for state in training_aggregates:
        train_agg = training_aggregates[state]
        val_agg = validation_aggregates.get(state, {})
        
        all_activities = set(train_agg.keys()) | set(val_agg.keys())
        vec_train = [train_agg.get(act, 0.0) for act in all_activities]
        vec_val = [val_agg.get(act, 0.0) for act in all_activities]
        
        dot_product = sum(a*b for a,b in zip(vec_train, vec_val))
        norm_train = math.sqrt(sum(a**2 for a in vec_train))
        norm_val = math.sqrt(sum(b**2 for b in vec_val))
        similarity = dot_product / (norm_train*norm_val) if norm_train>0 and norm_val>0 else 0.0
        
        state_similarities[state] = similarity
    
    return state_similarities, training_aggregates
	
def extract_event_payload_distributions(event_logs):
    trace_state_payloads = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for state, log in event_logs.items():
        if log is None:
            continue

        for trace in log:
            trace_id = trace.attributes.get("concept:name")
            if trace_id is None:
                continue

            for event in trace:
                activity = event["concept:name"]
                payload = event.get("payload_size", 0.0)
                trace_state_payloads[trace_id][state][activity].append(payload)

    return trace_state_payloads

def cosine_similarity_per_state_payload(training_payloads, validation_payloads):
    training_aggregates = defaultdict(dict)
    training_counts = defaultdict(lambda: defaultdict(int))

    for trace_id, states in training_payloads.items():
        for state, activities in states.items():
            for activity, payloads in activities.items():
                total = sum(payloads)
                count = len(payloads)
                training_aggregates[state][activity] = training_aggregates[state].get(activity, 0.0) + total
                training_counts[state][activity] = training_counts[state].get(activity, 0) + count

    for state in training_aggregates:
        for activity in training_aggregates[state]:
            training_aggregates[state][activity] /= training_counts[state][activity]

    validation_aggregates = defaultdict(dict)
    validation_counts = defaultdict(lambda: defaultdict(int))

    for trace_id, states in validation_payloads.items():
        for state, activities in states.items():
            for activity, payloads in activities.items():
                validation_aggregates[state][activity] = validation_aggregates[state].get(activity, 0.0) + sum(payloads)
                validation_counts[state][activity] = validation_counts[state].get(activity, 0) + len(payloads)

    for state in validation_aggregates:
        for activity in validation_aggregates[state]:
            validation_aggregates[state][activity] /= validation_counts[state][activity]

    state_similarities = {}
    for state in training_aggregates:
        train_agg = training_aggregates[state]
        val_agg = validation_aggregates.get(state, {})

        all_activities = set(train_agg.keys()) | set(val_agg.keys())
        vec_train = [train_agg.get(act, 0.0) for act in all_activities]
        vec_val = [val_agg.get(act, 0.0) for act in all_activities]

        dot_product = sum(a*b for a,b in zip(vec_train, vec_val))
        norm_train = math.sqrt(sum(a**2 for a in vec_train))
        norm_val = math.sqrt(sum(b**2 for b in vec_val))
        similarity = dot_product / (norm_train*norm_val) if norm_train>0 and norm_val>0 else 0.0

        state_similarities[state] = similarity

    return state_similarities, training_aggregates


	


def compute_per_trace_fitness(petri_nets, event_logs):

	trace_fitness_map = defaultdict(list)

	for state, log in event_logs.items():

		if log is None or len(log) == 0:
			continue

		for trace in log:
			trace_id = trace.attributes.get("concept:name")

			if trace_id is None:
				continue

			single_trace_log = EventLog([trace])

			try:
				fitness, _ = compute_fitness(
					petri_nets[state],
					single_trace_log,
					cc_variant="ALIGNMENT_BASED"
				)
			except Exception:
				fitness, _ = compute_fitness(
					petri_nets[state],
					single_trace_log,
					cc_variant="TOKEN_BASED"
				)

			trace_fitness_map[trace_id].append(fitness)

	print("Number of unique traces:", len(trace_fitness_map))
	print("Avg states per trace:",
		  mean(len(v) for v in trace_fitness_map.values()))

	averaged_fitness_per_trace = [
		mean(fitness_list)
		for fitness_list in trace_fitness_map.values()
		if len(fitness_list) > 0
	]

	return averaged_fitness_per_trace
	
def save_metrics(duration_similarity, alignment_similarity):

	with open(output_metrics_dir + "duration_similarity.txt", "w") as f:
		f.write(f"duration_similarity: {duration_similarity}\n")

	with open(output_metrics_dir + "alignment_similarity.txt", "w") as f:
		f.write(f"alignment_similarity: {alignment_similarity}\n")

	return None


def save_petri_nets(petri_nets):

	for state in petri_nets:
		pnml_exporter.apply(petri_nets[state]["network"], petri_nets[state]["initial_marking"], output_petrinets_dir + state + ".pnml", final_marking = petri_nets[state]["final_marking"])
		
	return None

def save_fitness_threshold(validation_fitness_values):

	with open(output_metrics_dir + "fitness_threshold.txt", "w") as f:
		f.write(f"fitness_threshold: {robust_fitness_threshold(validation_fitness_values)}\n")
		
def robust_fitness_threshold(fitness_values, k=2):
    mu = np.mean(fitness_values)
    sigma = np.std(fitness_values)
    return max(0.0, mu - k * sigma)	

def save_statewise_similarity(
    timing_sim, timing_aggregates,
    payload_sim, payload_aggregates,
    control_flow_sim, control_flow_aggregates
):
    filepath = os.path.join(output_metrics_dir, "statewise_similarity.txt")

    with open(filepath, "w") as f:
        for state in sorted(control_flow_sim.keys() | timing_sim.keys() | payload_sim.keys()):
            f.write(f"=== {state} ===\n\n")

            f.write("Control-flow similarity:\n")
            f.write(f"  {control_flow_sim.get(state, 0.0):.6f}\n")
            f.write("Control-flow training aggregates:\n")
            for activity, value in control_flow_aggregates.get(state, {}).items():
                f.write(f"  {activity}: {value:.6f}\n")
            f.write("\n")

            f.write("Timing similarity:\n")
            f.write(f"  {timing_sim.get(state, 0.0):.6f}\n")
            f.write("Timing training aggregates (mean durations):\n")
            for activity, value in timing_aggregates.get(state, {}).items():
                f.write(f"  {activity}: {value:.6f}\n")
            f.write("\n")

            f.write("Payload similarity:\n")
            f.write(f"  {payload_sim.get(state, 0.0):.6f}\n")
            f.write("Payload training aggregates (total payloads):\n")
            for activity, value in payload_aggregates.get(state, {}).items():
                f.write(f"  {activity}: {value:.6f}\n")
            f.write("\n\n")

    print(f"[INFO] State-wise similarities saved to {filepath}")
		
	
try:
	variant = sys.argv[1]
	noise_threshold = float(sys.argv[2])
	validation_percentage = float(sys.argv[3])
except IndexError:
	print("Enter the right number of input arguments.")
	sys.exit()

event_logs = read_event_logs()
training_event_logs, validation_event_logs = split_event_logs(event_logs, validation_percentage)
petri_nets = process_discovery(training_event_logs, variant, noise_threshold)

# Timing characterization
training_event_time_distribution = extract_event_time_distributions(training_event_logs)
validation_event_time_distribution = extract_event_time_distributions(validation_event_logs)
timing_trace_state_similarities, timing_training_aggregates = cosine_similarity_per_state_timing(training_event_time_distribution, validation_event_time_distribution)

# Payload characterization
training_event_payloads = extract_event_payload_distributions(training_event_logs)
validation_event_payloads = extract_event_payload_distributions(validation_event_logs)
payload_trace_state_similarities, payload_training_aggregates = cosine_similarity_per_state_payload(training_event_payloads,validation_event_payloads)

# Control-flow characterization
training_event_alignments = compute_event_alignments(petri_nets, training_event_logs)
validation_event_alignments = compute_event_alignments(petri_nets, validation_event_logs)
control_flow_trace_state_similarities, control_flow_training_aggregates = cosine_similarity_per_state(training_event_alignments, validation_event_alignments)

save_petri_nets(petri_nets)
save_statewise_similarity(timing_trace_state_similarities, timing_training_aggregates, payload_trace_state_similarities, payload_training_aggregates, control_flow_trace_state_similarities, control_flow_training_aggregates)