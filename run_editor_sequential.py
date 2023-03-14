"""Runs the RARR editor on a JSONL file of claims.

Runs the agreement gate and editor on a file with claims using GPT-3 and Bing.

Example usage:
python run_editor_sequential.py \
    --input_file data/palm/nq.jsonl \
    --output_file output/nq_davinci.jsonl \
    --model text-davinci-002 \
"""
import argparse
import json
import os
from typing import Any, Dict

import jsonlines
import Levenshtein
import tqdm

from prompts import rarr_prompts
from utils import (
    agreement_gate,
    editor,
    evidence_selection,
    search,
    question_generation,
)


def gate_cache_key(claim: str, query: str, evidence: str) -> str:
    """Defines how the key of the agreement gate is defined.

    Args:
        claim: Claim.
        query: Query.
        evidence: Evidence.
    Returns:
        Key to the agreement gate cache
    """
    return claim + query + evidence


def run_editor_one_instance(
    claim: str,
    context: str = None,
    model: str = "text-davinci-003",
    temperature_qgen: float = 0.7,
    num_rounds_qgen: int = 3,
    max_search_results_per_query: int = 3,
    max_passages_per_search_result: int = 1,
    max_sentences_per_passage: int = 5,
    sliding_distance: int = 1,
    max_evidences_per_question: int = 1,
    max_edit_ratio: float = 100,
) -> Dict[str, Any]:
    """Runs query generation, search, agreement gating, and editing on a claim.

    Args:
        claim: Text to check the validity of.
        model: Name of the OpenAI GPT-3 model to use.
        num_rounds_qgen: Number of times to sample questions.
        max_search_results_per_query: Maximum number of search results per query.
        max_sentences_per_passage: Maximum number of sentences for each passage.
        sliding_distance: Sliding window distance over the sentences of each search
            result. Used to extract passages.
        max_passages_per_search_result:  Maximum number of passages to return for
            each search result. A passage ranker is applied first.
        max_evidences_per_question: Maximum number of evidences to return per question.
        cached_questions: Cached results of question generation.
        cached_evidences_for_questions: Cached results of evidence retrieval.
        cached_agreement_gates: Maps from a concatenation of the claim, query, and
            evidence to the agreement gate result.
    Returns:
        result: All revision information, including the queries generated, search
            results, agreement gate information and each revision step done on the
            claim.
    """
    original_claim = claim
    agreement_gates = []

    # Generate questions for the claim
    questions = question_generation.run_rarr_question_generation(
        context=context,
        claim=claim,
        model=model,
        prompt=rarr_prompts.QGEN_PROMPT,
        temperature=temperature_qgen,
        num_rounds=num_rounds_qgen,
    )

    # Run search on generated question for the claim
    evidences_for_questions = [
        search.run_search(
            query=query,
            max_search_results_per_query=max_search_results_per_query,
            max_sentences_per_passage=max_sentences_per_passage,
            sliding_distance=sliding_distance,
            max_passages_per_search_result_to_return=max_passages_per_search_result,
        )
        for query in questions
    ]

    # Iterate through all questions generated for the claim
    used_evidences = [
        e
        for cur_evids in evidences_for_questions
        for e in cur_evids[:max_evidences_per_question]
    ]

    # Iterative editing over each evidence
    revision_steps = []
    for evid in used_evidences:
        # Run the agreement gate on the current (claim, query, evidence) tuple
        gate = agreement_gate.run_agreement_gate(
            context=context,
            claim=claim,
            query=evid["query"],
            evidence=evid["text"],
            model=model,
            prompt=rarr_prompts.AGREEMENT_GATE_PROMPT,
        )
        agreement_gates.append(gate)

        # Run the editor gate if the agreement gate is open
        if gate["is_open"]:
            edited_claim = editor.run_rarr_editor(
                context=context,
                claim=claim,
                query=evid["query"],
                evidence=evid["text"],
                model=model,
                prompt=rarr_prompts.EDITOR_PROMPT,
            )["text"]
            if Levenshtein.distance(claim, edited_claim) / len(claim) <= max_edit_ratio:
                claim = edited_claim
        revision_steps.append({"text": claim})

    result = {
        "text": original_claim,
        "questions": questions,
        "evidences_for_questions": evidences_for_questions,
        "revisions": [
            {
                "original_text": original_claim,
                "revised_text": revision_steps[-1]["text"],
                "evidences": used_evidences,
                "agreement_gates": agreement_gates,
                "revision_steps": revision_steps,
            }
        ],
    }
    selected_evidences = evidence_selection.select_evidences(result)
    result["selected_evidences"] = selected_evidences
    return result


def get_args() -> argparse.Namespace:
    """Gets command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="JSONLines file of claims to run RARR on.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="JSONLines file to write revisions to.",
    )
    parser.add_argument(
        "--claim_field",
        default="model_outputs_explanation",
        type=str,
        help="Field of the JSONL file to run the claim editing on.",
    )
    parser.add_argument(
        "--model",
        default="text-davinci-003",
        type=str,
        help="OpenAI GPT-3 model to use.",
    )
    parser.add_argument(
        "--temperature_qgen",
        default=0.7,
        type=float,
        help="Sampling temperature to use for query generation.",
    )
    parser.add_argument(
        "--num_rounds_qgen",
        default=3,
        type=int,
        help="Number of times to re-sample queries for a claim.",
    )
    parser.add_argument(
        "--max_search_results_per_query",
        default=3,
        type=int,
        help="Maximum number of search results we get per query.",
    )
    parser.add_argument(
        "--max_passages_per_search_result",
        default=1,
        type=int,
        help="Maximum number of passages to return for each search result. A passage"
        " ranker is applied to get the top passages per query.",
    )
    parser.add_argument(
        "--max_sentences_per_passage",
        default=5,
        type=int,
        help="Maximum number of sentences per evidence passage.",
    )
    parser.add_argument(
        "--sliding_distance",
        default=1,
        type=int,
        help="Sliding window distance for extracting passages from a search result.",
    )
    parser.add_argument(
        "--max_evidences_per_question",
        default=1,
        type=int,
        help="Maximum number of evidences to consider per question.",
    )
    parser.add_argument(
        "--max_edit_ratio",
        default=100,
        type=float,
        help="Maximum edit ratio between claim and edit for each round.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resumes the editing process if broken by loading the output file.",
    )
    args = parser.parse_args()

    # Write all args to file
    with open(args.output_file + "_args", "w", encoding="utf-8") as writer:
        json.dump(args.__dict__, writer, indent=4)
    return args


def main() -> None:
    """Loads a RARR evaluation set and runs GPT-3 RARR editing."""
    args = get_args()

    # Load the finished results by mapping from the claim name to the results.
    if args.resume and os.path.exists(args.output_file):
        print(f"Resuming with results from {args.output_file}")
        finished_results = {
            l["input_info"][args.claim_field]: l["result"]
            for l in jsonlines.open(args.output_file)
        }
        print(f"Found {len(finished_results)} finished lines.")
    else:
        finished_results = None

    with open(args.output_file, "w", encoding="utf-8") as writer:
        lines = list(jsonlines.open(args.input_file))
        for line in tqdm.tqdm(lines):
            claim = line["input_info"][args.claim_field]

            # Search for finished result
            if finished_results and claim in finished_results:
                line["result"] = finished_results[claim]
            else:
                line["result"] = run_editor_one_instance(
                    model=args.model,
                    claim=claim,
                    temperature_qgen=args.temperature_qgen,
                    num_rounds_qgen=args.num_rounds_qgen,
                    max_search_results_per_query=args.max_search_results_per_query,
                    max_passages_per_search_result=args.max_passages_per_search_result,
                    max_sentences_per_passage=args.max_sentences_per_passage,
                    sliding_distance=args.sliding_distance,
                    max_evidences_per_question=args.max_evidences_per_question,
                    max_edit_ratio=args.max_edit_ratio,
                )
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
