import json
import os
from datasets import load_dataset
from schemas.fhir_input import PatientCase, Observation

def load_and_cache_hf_cases(num_cases=5, output_dir="data/synthetic"):
    """
    Loads clinical notes from Hugging Face and maps them to our internal PatientCase schema.
    Since 'dlyog/synth_clin_notes' is mostly text-based, structured fields are mocked
    with placeholders, while the actual clinical note is passed in the `notes` field.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading {num_cases} cases from 'dlyog/synth_clin_notes'...")
    ds = load_dataset("dlyog/synth_clin_notes", split="train")
    
    cases = []
    for i in range(num_cases):
        record = ds[i]
        
        # Fallback to different possible text column names
        text = record.get("text", record.get("note", str(record)))
        
        # Map to our PatientCase schema
        case = PatientCase(
            case_id=f"HF-SYNTH-{i+1:03d}",
            age=50,  # Placeholder
            sex="Unknown",  # Placeholder
            chief_complaint="Review clinical notes for chief complaint",
            symptoms=["Refer to notes"],
            observations=[
                # Placeholder observation so the schema is valid
                Observation(code="N/A", display="Placeholder Observation", value="0")
            ],
            notes=[text]
        )
        
        cases.append(case)
        
        out_path = os.path.join(output_dir, f"{case.case_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(case.model_dump_json(indent=2))
            
    print(f"Saved {num_cases} parsed cases to {output_dir}")
    return cases

if __name__ == "__main__":
    load_and_cache_hf_cases()
