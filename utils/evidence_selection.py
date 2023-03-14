import itertools
from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

PASSAGE_RANKER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device=7
)


def get_unique_questions(example) -> List[str]:
    """Returns a list of unique question strings."""
    return sorted(set(example["questions"]))


def get_unique_evidences(example) -> List[str]:
    """Returns a list of unique evidence strings."""
    evidences = example["revisions"][0]["evidences"]
    return sorted(set(evidence["text"] for evidence in evidences))


def compute_score_matrix(
    questions: List[str], evidences: List[str]
) -> List[List[float]]:
    """Scores the questions against the evidences."""
    score_matrix = []
    for q in questions:
        evidence_scores = PASSAGE_RANKER.predict([(q, e) for e in evidences]).tolist()
        score_matrix.append(evidence_scores)
    return score_matrix


def question_coverage_objective_fn(
    score_matrix: List[List[float]], evidence_indices: List[int]
) -> float:
    """Returns the question coverage score."""
    # Compute sum_{question q} max_{selected evidence e} score(q, e).
    # This encourages all questions to be explained by at least one evidence.
    total = 0.0
    for scores_for_question in score_matrix:
        total += max(scores_for_question[j] for j in evidence_indices)
    return total


def select_evidences(
    example, prefer_fewer: bool = False, max_selected: int = 5
) -> List[Dict[str, Any]]:
    """Selects the set of evidences that maximizes the objective function.

    Args:
    score_matrix: A 2D list of size num_questions x num_evidences.
    max_selected: Maximum number of evidences to select.
    prefer_fewer: If True and the maximum objective value can be achieved by
      fewer evidences than max_selected, prefer selecting fewer evidences.

    Returns:
        selected_evidences: Selected evidences.
    """
    questions = get_unique_questions(example)
    evidences = get_unique_evidences(example)
    score_matrix = compute_score_matrix(questions, evidences)
    num_evidences = len(evidences)
    if num_evidences == 0:
        return {}

    best_combo = tuple()
    best_objective_value = float("-inf")
    max_selected = min(max_selected, num_evidences)
    min_selected = 1 if prefer_fewer else max_selected
    for num_selected in range(min_selected, max_selected + 1):
        for combo in itertools.combinations(range(num_evidences), num_selected):
            objective_value = question_coverage_objective_fn(score_matrix, combo)
            if objective_value > best_objective_value:
                best_combo = combo
                best_objective_value = objective_value

    selected_evidences = [{"text": evidences[idx]} for idx in best_combo]
    return selected_evidences
