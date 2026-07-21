import os
import sys
import json
import logging
from pprint import pprint

# Ensure the root project directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas.fhir_input import PatientCase
from agents.primary_llm.loop import run_grounding_loop

logging.basicConfig(level=logging.INFO)

def run_evaluation():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
    
    # Get all JSON files in the synthetic data directory
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist. Run hf_loader.py first.")
        return
        
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    
    if not json_files:
        print("No synthetic cases found. Run hf_loader.py first.")
        return
        
    # Pick the first case for evaluation
    test_file = os.path.join(data_dir, json_files[0])
    print(f"Evaluating Milestone 1 against: {test_file}")
    
    with open(test_file, "r", encoding="utf-8") as f:
        case_data = json.load(f)
        
    case = PatientCase(**case_data)
    
    # We will pass 'llama3' or a tiny model if 'openbiollm' is not available,
    # but the generate.py defaults to 'openbiollm' unless specified. 
    # Let's use whatever Ollama has available, or the default.
    model_override = os.getenv("OLLAMA_MODEL")
    
    print(f"\n--- Running Grounding Loop for Case {case.case_id} ---")
    final_hypothesis, audit_log = run_grounding_loop(case, model=model_override, max_iterations=3)
    
    print("\n=== FINAL HYPOTHESIS ===")
    print(final_hypothesis.hypothesis_text)
    
    print("\n=== CLAIMS & GROUNDING STATUS ===")
    for claim in final_hypothesis.claims:
        status = "✅ GROUNDED" if claim.grounded else "❌ UNGROUNDED"
        print(f"[{status}] {claim.text}")
        print(f"    Ref: {claim.supporting_evidence_ref}")
        if claim.grounding_reason:
            print(f"    Reason: {claim.grounding_reason}")
            
    print("\n=== OVERALL RESULT ===")
    print(f"Fully Grounded: {final_hypothesis.is_fully_grounded}")
    print(f"Iterations taken: {len(audit_log)}")

if __name__ == "__main__":
    run_evaluation()
