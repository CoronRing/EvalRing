"""
Simplified script to reproduce CounselBench research with 3 random samples.
This script evaluates LLM responses on mental health counseling questions.
"""
import sys
import os
import json
import random
from datetime import datetime
from tqdm import tqdm

# Add the CounselBench directory to the path
# Relative path from examples/suicide_detection/ground_truth to the repository root
counselbench_rel_path = os.path.join('..', '..', '..', 'CounselBench')
counselbench_path = os.path.abspath(os.path.join(os.path.dirname(__file__), counselbench_rel_path))

print(f"CounselBench path: {counselbench_path}")
sys.path.insert(0, counselbench_path)

try:
    from models.openai_llm import OpenAIModel
except ImportError as e:
    print(f"Error importing from CounselBench: {e}")
    # Try importing directly if path is weird
    try:
        sys.path.append(counselbench_path)
        from models.openai_llm import OpenAIModel
    except ImportError:
        print("Could not import models even after appending path.")
        sys.exit(1)

def setup_output_dir():
    """Create output directory for results"""
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'results')
    logs_dir = os.path.join(base_dir, 'logs')
    data_dir = os.path.join(base_dir, 'data')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    return output_dir, logs_dir, data_dir

def load_prepared_samples():
    """Load the 3 prepared samples from CounselBench data directory"""
    # Path to the prepared samples file in CounselBench directory
    samples_file_path = os.path.join(counselbench_path, 'data', 'counsel_chat', '3_sample_test.json')
    
    print(f"Loading samples from: {samples_file_path}")
    
    if not os.path.exists(samples_file_path):
        print(f"Error: Samples file not found at {samples_file_path}")
        print("Please run prepare_3_samples.py in the CounselBench directory first.")
        # Fallback to creating dummy samples if file doesn't exist to prevent crash
        return [{
            'questionID': 0, 'topic': 'test', 'questionTitle': 'Test Q', 
            'questionText': 'Test Text', 'answerText': 'Test Answer', 'upvotes': 0
        }]
        
    with open(samples_file_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
        
    # Add index if missing
    for i, sample in enumerate(samples):
        if 'index' not in sample:
            sample['index'] = i
            
    print(f"Loaded {len(samples)} samples.")
    return samples

def generate_responses(samples, model_name="gpt-4o", temperature=0.7):
    """Generate LLM responses for the sampled questions"""
    print(f"\nGenerating responses using {model_name}...")
    
    # Initialize model
    # Note: OpenAIModel will look for config.json in the current working directory
    try:
        model = OpenAIModel(
            model_name=model_name,
            temperature=temperature,
            task_name="counsel_chat",
            prompt_name="persona_survey",
            is_length_constrained=True
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Make sure config.json is present in the current directory and contains valid API keys.")
        sys.exit(1)
    
    results = []
    
    for sample in tqdm(samples, desc="Generating responses"):
        # Construct input text
        if sample.get('questionText') is None or sample.get('questionText', '').strip() == '':
            input_text = sample.get('questionTitle', '')
        else:
            input_text = f"{sample.get('questionTitle', '')} {sample.get('questionText', '')}"
        
        # Generate response
        try:
            response, generate_count = model.regenerate_until_valid_length(input_text)
            word_count = len(response.split())
            
            result = {
                'index': sample.get('index', 0),
                'questionID': sample.get('questionID'),
                'topic': sample.get('topic'),
                'input_text': input_text,
                'original_answer': sample.get('answerText'),
                'original_upvotes': sample.get('upvotes'),
                'llm_response': response,
                'word_count': word_count,
                'generate_attempts': generate_count,
                'model_name': model_name,
                'temperature': temperature
            }
            
            results.append(result)
            
            print(f"\n--- Question (ID: {sample.get('questionID')}) ---")
            print(f"Topic: {sample.get('topic')}")
            print(f"Input: {input_text[:100]}...")
            print(f"Response: {response[:100]}...")
            print(f"Word count: {word_count}, Attempts: {generate_count}")
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            results.append({
                'index': sample.get('index', 0),
                'questionID': sample.get('questionID'),
                'error': str(e)
            })
    
    return results

def save_results(samples, results, output_dir, logs_dir, data_dir):
    """Save evaluation results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw samples copy
    samples_file = os.path.join(data_dir, f'samples_{timestamp}.json')
    with open(samples_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"\nSaved samples copy to: {samples_file}")
    
    # Save evaluation results
    results_file = os.path.join(output_dir, f'evaluation_results_{timestamp}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to: {results_file}")
    
    # Create a summary log
    log_file = os.path.join(logs_dir, f'evaluation_log_{timestamp}.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== CounselBench Evaluation Log ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of samples: {len(samples)}\n")
        if results:
            f.write(f"Model: {results[0].get('model_name', 'unknown')}\n")
            f.write(f"Temperature: {results[0].get('temperature', 'unknown')}\n")
        
        f.write("\n=== Results Summary ===\n")
        
        for i, result in enumerate(results):
            f.write(f"\n--- Sample {i+1} ---\n")
            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            else:
                f.write(f"Question ID: {result.get('questionID')}\n")
                f.write(f"Topic: {result.get('topic')}\n")
                f.write(f"Original upvotes: {result.get('original_upvotes')}\n")
                f.write(f"Generated word count: {result.get('word_count')}\n")
                f.write(f"Generation attempts: {result.get('generate_attempts')}\n")
    
    print(f"Saved log to: {log_file}")
    
    # Save a comparison file
    comparison_file = os.path.join(output_dir, f'comparison_{timestamp}.txt')
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("=== CounselBench Response Comparison ===\n\n")
        
        for i, result in enumerate(results):
            if 'error' not in result:
                f.write(f"{'='*80}\n")
                f.write(f"QUESTION {i+1} (ID: {result.get('questionID')}, Topic: {result.get('topic')})\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"INPUT:\n{result.get('input_text')}\n\n")
                f.write(f"ORIGINAL THERAPIST RESPONSE (upvotes: {result.get('original_upvotes')}):\n")
                f.write(f"{result.get('original_answer')}\n\n")
                f.write(f"LLM RESPONSE ({result.get('model_name')}, {result.get('word_count')} words):\n")
                f.write(f"{result.get('llm_response')}\n\n")
    
    print(f"Saved comparison to: {comparison_file}")

def main():
    """Main execution function"""
    print("="*80)
    print("CounselBench Research Reproduction")
    print("Evaluating LLM responses on mental health counseling")
    print("="*80)
    
    setup_output_dir()
    output_dir, logs_dir, data_dir = setup_output_dir()
    
    # Load samples
    samples = load_prepared_samples()
    
    # Generate responses
    results = generate_responses(samples, model_name="gpt-4o", temperature=0.7)
    
    # Save results
    save_results(samples, results, output_dir, logs_dir, data_dir)
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
