from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph


print("Extension 2", "-" * 40)


class ReviewDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


@dataclass
class LoanRequest:
    application_id: str
    loan_amount: float
    applicant_summary: str
    ai_recommendation: str
    risk_score: float
    review_required: bool = False
    human_decision: Optional[ReviewDecision] = None
    reviewer_id: Optional[str] = None
    review_notes: Optional[str] = None


class LoanWorkflowState(TypedDict):
    request: LoanRequest
    route: str
    final_message: str


class HITLWorkflow:
    """
    Human-in-the-loop workflow built with LangGraph.
    High-value or high-risk loans pause for human approval.
    """

    HIGH_VALUE_THRESHOLD = 1_000_000  # INR 10 lakhs
    HIGH_RISK_SCORE = 0.7

    def __init__(self) -> None:
        self.review_queue: List[LoanRequest] = []
        self.app = self._build_app()

    def _route_request(self, state: LoanWorkflowState) -> LoanWorkflowState:
        request = state["request"]
        needs_review = (
            request.loan_amount >= self.HIGH_VALUE_THRESHOLD
            or request.risk_score >= self.HIGH_RISK_SCORE
        )

        request.review_required = needs_review
        route = "human_review" if needs_review else "auto_process"

        if needs_review and request not in self.review_queue:
            self.review_queue.append(request)

        return {**state, "route": route}

    def _auto_process(self, state: LoanWorkflowState) -> LoanWorkflowState:
        request = state["request"]
        request.human_decision = ReviewDecision.APPROVE
        request.reviewer_id = "SYSTEM_AUTO"
        request.review_notes = "Auto-approved by policy because human review was not required."
        return {
            **state,
            "final_message": (
                f"[AUTO] {request.application_id} approved automatically "
                f"for INR {request.loan_amount:,.0f}."
            ),
        }

    def _human_review(self, state: LoanWorkflowState) -> LoanWorkflowState:
        request = state["request"]
        decision = (
            ReviewDecision.APPROVE
            if request.risk_score < self.HIGH_RISK_SCORE
            else ReviewDecision.REJECT
        )
        reviewer_id = "UNDERWRITER_001"
        notes = "Manual review complete."

        request.human_decision = decision
        request.reviewer_id = reviewer_id
        request.review_notes = notes

        if request in self.review_queue:
            self.review_queue.remove(request)

        return {
            **state,
            "final_message": (
                f"[HUMAN] {request.application_id} {decision.value.upper()} "
                f"by {reviewer_id}."
            ),
        }

    def _route_after_assessment(self, state: LoanWorkflowState) -> str:
        return state["route"]

    def _build_app(self):
        graph = StateGraph(LoanWorkflowState)

        graph.add_node("route_request", self._route_request)
        graph.add_node("auto_process", self._auto_process)
        graph.add_node("human_review", self._human_review)

        graph.add_conditional_edges(
            "route_request",
            self._route_after_assessment,
            {
                "auto_process": "auto_process",
                "human_review": "human_review",
            },
        )

        graph.add_edge("auto_process", END)
        graph.add_edge("human_review", END)
        graph.set_entry_point("route_request")
        return graph.compile()

    def run(self, request: LoanRequest) -> LoanWorkflowState:
        initial_state: LoanWorkflowState = {
            "request": request,
            "route": "",
            "final_message": "",
        }
        return self.app.invoke(initial_state)

    def queue_status(self) -> None:
        print(f"\nREVIEW QUEUE STATUS: {len(self.review_queue)} pending")
        for request in self.review_queue:
            print(
                f"   [{request.application_id}] INR {request.loan_amount:,.0f} | "
                f"Risk: {request.risk_score:.2f} | "
                f"{request.applicant_summary[:50]}"
            )


def main() -> None:
    workflow = HITLWorkflow()

    test_loans = [
        LoanRequest(
            "APP100301",
            500_000,
            "Salaried, credit score 720, DTI 30%",
            "Likely approve",
            0.25,
        ),
        LoanRequest(
            "APP100302",
            1_500_000,
            "Self-employed, credit score 660, DTI 45%",
            "Borderline",
            0.62,
        ),
        LoanRequest(
            "APP100303",
            200_000,
            "Salaried, credit score 580, DTI 55%",
            "High risk",
            0.82,
        ),
        LoanRequest(
            "APP100304",
            800_000,
            "Salaried, credit score 750, DTI 25%",
            "Likely approve",
            0.18,
        ),
    ]

    print("\nLANGGRAPH-STYLE HITL ROUTING")
    print("-" * 65)
    for loan in test_loans:
        final_state = workflow.run(loan)
        icon = "[HR]" if final_state["route"] == "human_review" else "[AUTO]"
        print(
            f"{icon} [{final_state['route'].upper():14}] "
            f"[{loan.application_id}] INR {loan.loan_amount:>12,.0f} | "
            f"Risk: {loan.risk_score:.2f}"
        )
        print(f"   {final_state['final_message']}")

    workflow.queue_status()


if __name__ == "__main__":
    main()
