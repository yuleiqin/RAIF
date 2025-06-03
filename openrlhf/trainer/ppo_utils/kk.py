import re
from typing import Dict, Tuple, Optional
import json


def extract_solution(solution_str: str, template:str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    processed_str = solution_str
    # Extract think content using XML-style tags
    think_pattern = r'<think>(.*?)</think>'
    matches_think = list(re.finditer(think_pattern, processed_str, re.DOTALL))

    if matches_think:
        final_think = matches_think[-1].group(1).strip()
    else:
        print("[Error] No valid think tags found")
        final_think = None

    if template == "deepseek":
        if "</think>" in processed_str:
            final_answer = (processed_str[processed_str.rfind("</think>")+len("</think>"):]).strip()
        else:
            final_answer = None
    
    else:
        # Extract final answer using XML-style tags
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

        if matches:
            final_answer = matches[-1].group(1).strip()
        else:
            print("[Error] No valid answer tags found")
            final_answer = None

    return final_answer, final_think, processed_str


def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.
    
    Args:
        solution_text: Formatted solution text from dataset
        
    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    print("\n[Ground Truth Parsing]")
    
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")
    
    return status_dict



def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification
        
    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    print("\n[Model Answer Parsing]")
    print(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count('knight')
    knave_count = answer_text.lower().count('knave')

    print(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        print(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
        return None

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Error] Missing identification for {name}")
            return None
    
    return status_dict




def validate_response_structure(processed_str: str, template: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    if template == "deepseek":
        # Check required tags
        tags = {
            'think_start': ('<think>', 1),
            'think_end': ('</think>', 1),
        }

        positions = {}
        for tag_name, (tag_str, expected_count) in tags.items():
            count = processed_str.count(tag_str)
            positions[tag_name] = pos = processed_str.find(tag_str)
            print(f"  {tag_str}: count={count}, position={pos}")
            if count != expected_count:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
                validation_passed = False

        # Verify tag order
        if (positions['think_start'] > positions['think_end']):
            print("  [Error] Incorrect tag order: Expected <think>...</think>")
            validation_passed = False
        else:
            print("  Tag sequence validation passed")
    
    else:
        # Check required tags
        tags = {
            'think_start': ('<think>', 1),
            'think_end': ('</think>', 1),
            'answer_start': ('<answer>', 1),
            'answer_end': ('</answer>', 1)
        }

        positions = {}
        for tag_name, (tag_str, expected_count) in tags.items():
            count = processed_str.count(tag_str)
            positions[tag_name] = pos = processed_str.find(tag_str)
            
            print(f"  {tag_str}: count={count}, position={pos}")
            
            if count != expected_count:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
                validation_passed = False

        # Verify tag order
        if (positions['think_start'] > positions['think_end'] or
            positions['think_end'] > positions['answer_start'] or
            positions['answer_start'] > positions['answer_end']):
            print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
            validation_passed = False
        else:
            print("  Tag sequence validation passed")

    return validation_passed


def compute_score(solution_str: str,
                #  ground_truth: Dict[str, str],
                ground_truth: str,
                template: str,
                format_reward: float = 1.0,
                answer_reward: float = 2.0,
                ignore_format: bool = False):
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    
    # Parse ground truth data
    ground_truth = json.loads(ground_truth)

    solution_text = ground_truth.get('solution_text_format', '')
    gt_status = parse_solution_text_format(solution_text)
    expected_names = list(gt_status.keys())
    print(f"[Ground Truth] Final identities: {gt_status}")

    # Extract model answer
    if ignore_format:
        format_correct = True
        answer_text = solution_str
        think_text = solution_str
        processed_str = solution_str
        print("---Ignore Formatting---")

    else:
        answer_text, think_text, processed_str = extract_solution(solution_str, template=template)
        # Validate response structure
        format_correct = validate_response_structure(processed_str, template=template)

    print(f"\n[Model Response]\n{processed_str}")
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    # Validate answer content
    answer_score = 0.0
    is_answer_correct = 0
    if format_correct and answer_text and (think_text and think_text.lower() != "reasoning process here"):
        pred_status = parse_model_answer(answer_text, expected_names)
        if pred_status:
            print(f"\n[Content Validation]")
            print(f"  Expected: {gt_status}")
            print(f"  Predicted: {pred_status}")
            if pred_status == gt_status:
                answer_score = answer_reward
                is_answer_correct = 1
                print("  Content validation: FULL MATCH")
            else:
                answer_score = -abs(0.75*answer_reward)
                print("  Content validation: MISMATCH")
        else:
            answer_score = -abs(answer_reward)
            print( "Fail to parse answer")
    else:
        answer_score = -abs(answer_reward)
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    # # total_score = format_score + answer_score
    # print("\n" + "-"*80)
    # print(f" Final Score ".center(80, '-'))
    # print(f"  Format: {format_score}")
    # print(f"  Answer: {answer_score}")
    # print(f"  Total: {total_score}")
    # print("="*80 + "\n")

    return answer_score, is_answer_correct
