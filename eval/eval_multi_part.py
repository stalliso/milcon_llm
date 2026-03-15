# eval/eval_multi_part.py
#
# USAGE:
#   from eval.eval_multi_part import eval_multi_part_routing
#   eval_multi_part_routing(app)

import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# =============================================================================
#  QUESTION BANK
# =============================================================================

MULTI_PART_QUESTIONS = {

    # proj + strat26 + strat28
    "Q1_fire_water_pom26_pom28": (
        "What is the CWE and project scope for RM21-0774? "
        "What Severity/Urgency score did it receive under POM26 criteria, and what was the stated justification? "
        "How does the Mission Alignment definition differ between POM26 and POM28 scoring guidance? "
        "Based on the updated POM28 criteria, how would you expect the Mission Alignment score for RM21-0774 to change, and why?"
    ),

    # proj + strat26
    "Q2_airfield_pom26_alignment": (
        "What is the installation, COCOM, and CWE for project ST18-1369? "
        "Summarize the project scope and the stated impact if not provided. "
        "What Readiness Support and Operational Cost scores were assigned to this project, and what reasoning was given? "
        "How do the POM26 NDS strategic themes support the prioritization of airfield pavement repair projects like ST18-1369?"
    ),

    # proj + strat28
    "Q3_port_ops_warehouse_pom28_rescore": (
        "What is the scope and CWE for project NF20-0826? "
        "What were the lead proponent's Readiness Support and Urgency scores? "
        "What updated NDS/NSS strategic themes does POM28 introduce that are relevant to shore logistics and port operations infrastructure? "
        "Given POM28's updated NDS/NSS strategic priorities around logistics infrastructure, would Operational Cost remain a strong justification for NF20-0826?"
    ),

    # proj + strat26 + strat28
    "Q4_galley_ii_strategy_comparison": (
        "What is the mission of the facility for project P930 and what COCOM does it support? "
        "Summarize the Readiness Support and Mission Alignment scores assigned under POM26 and the proponent's justifications. "
        "How does the POM26 NSS guidance define Mission Alignment for dining/quality-of-life facilities in forward operating areas? "
        "Under POM28 updated criteria, would a galley expansion at a joint-use forward operating base like Camp Lemonnier score higher or lower on Mission Alignment — and what factors drive that assessment?"
    ),

    # proj only
    "Q5_aimd_hangar_project_details": (
        "What buildings are included in the scope of project RM16-0799 and what installation is it located at? "
        "List the deficiency codes and their severity ratings identified in the facility information for this project. "
        "What is the total SMS Requirement cost reported for RM16-0799? "
        "What impact does the project state would result if repairs are not provided?"
    ),

    # proj + strat26
    "Q6_air_freight_terminal_pom26": (
        "What is the project scope and CWE for RM17-0117 at NAS Sigonella? "
        "What Operational Cost and Readiness Support scores did the lead proponent assign, and what justifications were provided? "
        "What POM26 CNIC scoring criteria govern the Operational Cost category, and what payback thresholds determine each score level? "
        "How well does RM17-0117's documented payback period and safety justification align with the POM26 scoring rubric for Operational Cost?"
    ),

    # proj + strat28
    "Q7_beq_perm_party_pom28": (
        "What is the scope, CWE, and installation for project P1616? "
        "What force structure growth is cited in the project justification, and which squadrons or units are driving the requirement? "
        "How does the POM28 updated guidance define Readiness Support for unaccompanied housing projects supporting permanently assigned personnel? "
        "Based on the documented bed occupancy rate and Interim Assignment Policy in the P1616 justification, estimate what Readiness Support score this project would receive under POM28 criteria."
    ),

    # proj + strat26 + strat28
    "Q8_wastewater_souda_bay_full": (
        "What is the CWE, CCN, and project scope for RM20-0669 at NSA Souda Bay? "
        "What Operational Cost score did the proponent assign and what annual service value did they cite to support it? "
        "How does POM26 guidance define the Severity/Urgency category, and what score thresholds apply to environmental compliance and capacity loss scenarios? "
        "Considering the 50% capacity loss and documented environmental leaching risk, re-evaluate RM20-0669's Severity/Urgency score under POM28 definitions and explain any change."
    ),

    # proj only
    "Q9_air_cargo_terminal_attributes": (
        "What is the total CWE and project CCN for P942 at Camp Lemonnier? "
        "Which COCOMs does the Combined Air Cargo/Passenger Terminal directly support according to the mission statement? "
        "What is the square footage of the facility being constructed and what functional areas does it include? "
        "What impact does the project state would occur if the terminal is not provided?"
    ),

    # strat26 + strat28 only
    "Q10_scoring_rubric_pom26_vs_pom28": (
        "How does POM26 guidance define the four primary PDS scoring categories: Mission Alignment, Readiness Support, Operational Cost, and Severity/Urgency? "
        "What are the numerical score levels and the criteria that distinguish each level within the Severity/Urgency category under POM26? "
        "What changes did POM28 introduce to the Readiness Support scoring definitions compared to POM26? "
        "Which of the four scoring categories saw the most significant revision between POM26 and POM28 guidance, and what strategic rationale explains that change?"
    ),

    # proj + strat26
    "Q11_lift_station_rehab_pom26": (
        "What is the project scope and CWE for RM21-0395 at NAVSTA Rota? "
        "What is the mission of the lift station facility, and what failure modes does the project justification describe? "
        "What Operational Cost and Severity/Urgency scores were assigned to this project, and how are they justified? "
        "How does POM26 NDS strategy prioritize utility resilience and infrastructure recapitalization projects relative to operational readiness?"
    ),

    # proj + strat28
    "Q12_energy_resilience_pom28": (
        "What is the scope and CWE for project P1413 at NAVSTA Rota, and which utility systems does it integrate? "
        "What COCOM does this project support and what is the lead proponent listed? "
        "How does POM28 updated guidance treat energy resilience and cyber security of industrial control systems in its Mission Alignment scoring criteria? "
        "Would P1413 likely receive a higher Mission Alignment score under POM28 than POM26, given the updated emphasis on cyber resilience — and what specific POM28 language supports that assessment?"
    ),

    # proj + strat26 + strat28
    "Q13_airfield_apron_full_comparison": (
        "What is the PCI rating, CWE, and RAC level for project ST19-0946? "
        "What Readiness Support score did the lead proponent assign and what flight operations impact was cited? "
        "Under POM26 CNIC scoring criteria, what Readiness Support score threshold applies to projects that enable FDNF squadron readiness and prevent mission degradation? "
        "How would the Readiness Support and Severity/Urgency scores for ST19-0946 be evaluated under POM28 guidance, and would the overall project priority likely change?"
    ),

    # proj + strat26
    "Q14_air_passenger_terminal_pom26": (
        "What is the scope, CWE, and project CCN for RM17-1027 at NAS Sigonella? "
        "What capacity shortfall is documented in the facility information, and how many passengers per peak hour would the upgrade accommodate? "
        "What Mission Alignment and Readiness Support scores were assigned, and what multi-COCOM justification was provided? "
        "Under POM26 NSS/NDS themes, how is multi-COCOM joint-use facility support weighted in the Mission Alignment scoring rubric?"
    ),

    # proj + strat26 + strat28
    "Q15_arm_dearm_berm_full": (
        "What is the CWE, installation, and COCOM for project NF23-0892, and what is the primary mission the facility supports? "
        "What Operational Cost score and urgency justification did the lead proponent provide, including the TDY cost workaround described? "
        "Under POM26 scoring criteria, what Severity/Urgency score would a project with an active RAC and documented aircrew qualification degradation typically receive? "
        "Under POM28 updated guidance, how would changes to the Readiness Support definitions affect the scoring of a project that directly enables FDNF MH-60R aircrew weapons qualification at a forward-deployed installation?"
    ),
}


# =============================================================================
#  EXPECTED ROUTES
# =============================================================================

EXPECTED_ROUTES = {
    "Q1_fire_water_pom26_pom28":          ["proj_vectorstore", "strat26_vectorstore", "strat28_vectorstore"],
    "Q2_airfield_pom26_alignment":         ["proj_vectorstore", "strat26_vectorstore"],
    "Q3_port_ops_warehouse_pom28_rescore": ["proj_vectorstore", "strat28_vectorstore"],
    "Q4_galley_ii_strategy_comparison":    ["proj_vectorstore", "strat26_vectorstore", "strat28_vectorstore"],
    "Q5_aimd_hangar_project_details":      ["proj_vectorstore"],
    "Q6_air_freight_terminal_pom26":       ["proj_vectorstore", "strat26_vectorstore"],
    "Q7_beq_perm_party_pom28":             ["proj_vectorstore", "strat28_vectorstore"],
    "Q8_wastewater_souda_bay_full":        ["proj_vectorstore", "strat26_vectorstore", "strat28_vectorstore"],
    "Q9_air_cargo_terminal_attributes":    ["proj_vectorstore"],
    "Q10_scoring_rubric_pom26_vs_pom28":   ["strat26_vectorstore", "strat28_vectorstore"],
    "Q11_lift_station_rehab_pom26":        ["proj_vectorstore", "strat26_vectorstore"],
    "Q12_energy_resilience_pom28":         ["proj_vectorstore", "strat28_vectorstore"],
    "Q13_airfield_apron_full_comparison":  ["proj_vectorstore", "strat26_vectorstore", "strat28_vectorstore"],
    "Q14_air_passenger_terminal_pom26":    ["proj_vectorstore", "strat26_vectorstore"],
    "Q15_arm_dearm_berm_full":             ["proj_vectorstore", "strat26_vectorstore", "strat28_vectorstore"],
}


# =============================================================================
#  EVAL FUNCTION
# =============================================================================

def eval_multi_part_routing(app, k=6, verbose=False):
    """Run all multi-part routing tests against a compiled LangGraph app.

    Args:
        app:     Compiled LangGraph workflow.
        k:       Number of documents to retrieve per store.
        verbose: If True, print questions, routes, and answers.

    Returns:
        dict of per-question results: {expected, actual, passed, generation}
    """

    def run_one(item):
        q_key, question = item
        state         = app.invoke({"question": question, "k": k})
        actual_routes = state.get("routes", [])
        generation    = state.get("generation", "")
        passed        = sorted(actual_routes) == sorted(EXPECTED_ROUTES[q_key])
        return q_key, {
            "expected":   sorted(EXPECTED_ROUTES[q_key]),
            "actual":     sorted(actual_routes),
            "passed":     passed,
            "generation": generation,
        }

    # Limit concurrency to avoid overwhelming campus NPS API
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = dict(executor.map(run_one, MULTI_PART_QUESTIONS.items()))

    if verbose:
        print("\n" + "=" * 70)
        print("  MULTI-PART ROUTING EVALUATION")
        print("=" * 70)
        for q_key, res in results.items():
            print(f"\n{'-' * 70}")
            print(f"TEST : {q_key}")
            print(f"WANT : {res['expected']}")
            print(f"GOT  : {res['actual']}")
            print(f"PASS : {'✅ PASS' if res['passed'] else '❌ FAIL'}")
            print(f"\nANSWER:\n{res['generation']}")

        passed_count = sum(1 for v in results.values() if v["passed"])
        print("\n" + "=" * 70)
        print(f"  ROUTING ACCURACY:  {passed_count}/{len(results)} passed")
        print("=" * 70)
        for q_key, res in results.items():
            status = "✅" if res["passed"] else "❌"
            print(f"  {status}  {q_key}")
            if not res["passed"]:
                print(f"       Expected : {res['expected']}")
                print(f"       Got      : {res['actual']}")

    return results