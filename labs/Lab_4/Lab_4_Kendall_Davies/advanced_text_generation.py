#!/usr/bin/env python3
"""
Advanced Text Generation Testing for Sherlock Holmes RNN
This script provides additional testing functions for seeded text generation.
"""

import torch
import numpy as np
from torch.distributions import Categorical

def generate_text_with_seed(model, character_to_num, num_to_character, seed_phrase, length=200, temperature=1.0):
    """
    Generate text starting with a specific seed phrase
    
    Args:
        model: Trained RNN model
        character_to_num: Dictionary mapping characters to numbers
        num_to_character: Dictionary mapping numbers to characters
        seed_phrase: Starting text (e.g., "My dear fellow")
        length: Number of characters to generate
        temperature: Controls randomness (higher = more random)
    """
    model.eval()
    
    # Convert seed phrase to character indices
    seed_indices = []
    for char in seed_phrase:
        if char in character_to_num:
            seed_indices.append(character_to_num[char])
        else:
            print(f"Warning: Character '{char}' not found in vocabulary, skipping...")
            continue
    
    if not seed_indices:
        print("Error: No valid characters in seed phrase!")
        return ""
    
    # Convert to tensor and move to GPU
    seed_tensor = torch.tensor(seed_indices).unsqueeze(1).cuda()
    
    # Initialize hidden state
    hidden_state = None
    
    # Print the seed phrase
    print(f"Seed: '{seed_phrase}'")
    print("Generated text:")
    print(seed_phrase, end='')
    
    generated_text = seed_phrase
    
    # Use the seed phrase to initialize the model
    with torch.no_grad():
        for i in range(len(seed_indices) - 1):
            input_char = seed_tensor[i:i+1]
            output, hidden_state = model(input_char, hidden_state)
        
        # Now generate new text
        current_input = seed_tensor[-1:]  # Last character of seed
        
        for _ in range(length):
            output, hidden_state = model(current_input, hidden_state)
            
            # Apply temperature scaling
            output = output / temperature
            
            # Get probabilities
            probs = torch.nn.functional.softmax(torch.squeeze(output), dim=0)
            
            # Sample next character
            character_distribution = torch.distributions.Categorical(probs)
            character_num = character_distribution.sample()
            
            # Get the character
            char = num_to_character[character_num.item()]
            print(char, end='')
            generated_text += char
            
            # Update input for next iteration
            current_input = character_num.unsqueeze(0).unsqueeze(1)
    
    print("\n" + "="*50)
    return generated_text

def test_temperature_effects(model, character_to_num, num_to_character):
    """Test how temperature affects text generation"""
    print("Testing Temperature Effects:")
    print("="*60)
    
    seed_phrase = "My dear Watson"
    temperatures = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    
    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        print("-" * 30)
        generate_text_with_seed(model, character_to_num, num_to_character, 
                              seed_phrase, length=100, temperature=temp)

def test_seed_lengths(model, character_to_num, num_to_character):
    """Test how seed phrase length affects generation"""
    print("\nTesting Seed Phrase Lengths:")
    print("="*60)
    
    # Different length seed phrases
    seeds = {
        "Short (1-3 chars)": ["I", "He", "The"],
        "Medium (4-8 chars)": ["Holmes", "Watson", "Elementary"],
        "Long (9+ chars)": ["My dear fellow", "The game is afoot", "I have observed"]
    }
    
    for category, seed_list in seeds.items():
        print(f"\n{category}:")
        print("-" * 30)
        for seed in seed_list:
            generate_text_with_seed(model, character_to_num, num_to_character, 
                                  seed, length=80, temperature=0.8)

def test_sherlock_phrases(model, character_to_num, num_to_character):
    """Test with classic Sherlock Holmes phrases"""
    print("\nTesting Classic Sherlock Holmes Phrases:")
    print("="*60)
    
    classic_phrases = [
        "My dear fellow",
        "Elementary, my dear Watson",
        "The game is afoot",
        "I have observed",
        "It is quite simple",
        "The case presents",
        "Holmes said",
        "Watson, I have",
        "The evidence suggests",
        "I deduce that"
    ]
    
    for phrase in classic_phrases:
        generate_text_with_seed(model, character_to_num, num_to_character, 
                              phrase, length=120, temperature=0.8)

def interactive_generation(model, character_to_num, num_to_character):
    """Interactive text generation"""
    print("\nInteractive Text Generation:")
    print("="*60)
    print("Enter your own seed phrases (type 'quit' to exit):")
    
    while True:
        seed = input("\nEnter seed phrase: ").strip()
        if seed.lower() == 'quit':
            break
        
        if seed:
            try:
                length = int(input("Enter length (default 150): ") or "150")
                temp = float(input("Enter temperature (default 0.8): ") or "0.8")
                generate_text_with_seed(model, character_to_num, num_to_character, 
                                     seed, length=length, temperature=temp)
            except ValueError:
                print("Invalid input. Using defaults.")
                generate_text_with_seed(model, character_to_num, num_to_character, 
                                     seed, length=150, temperature=0.8)

def analyze_generation_quality(model, character_to_num, num_to_character, num_samples=5):
    """Analyze the quality of generated text"""
    print("\nAnalyzing Generation Quality:")
    print("="*60)
    
    test_phrases = ["My dear Watson", "The case was", "Holmes observed", "I have deduced", "Elementary"]
    
    for phrase in test_phrases:
        print(f"\nAnalyzing: '{phrase}'")
        print("-" * 40)
        
        # Generate multiple samples
        samples = []
        for i in range(num_samples):
            print(f"\nSample {i+1}:")
            text = generate_text_with_seed(model, character_to_num, num_to_character, 
                                         phrase, length=100, temperature=0.8)
            samples.append(text)
        
        # Basic analysis
        avg_length = sum(len(s) for s in samples) / len(samples)
        print(f"\nAverage length: {avg_length:.1f} characters")
        
        # Check for common words
        common_words = ['the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that']
        word_counts = {}
        for word in common_words:
            count = sum(s.lower().count(word) for s in samples)
            word_counts[word] = count
        
        print("Common word frequency:")
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  '{word}': {count}")

if __name__ == "__main__":
    print("This script provides testing functions for your RNN model.")
    print("To use these functions, you need to:")
    print("1. Load your trained model")
    print("2. Load your character mappings")
    print("3. Call the testing functions")
    print("\nExample usage:")
    print("test_temperature_effects(rnn, character_to_num, num_to_character)")
    print("test_sherlock_phrases(rnn, character_to_num, num_to_character)")
    print("interactive_generation(rnn, character_to_num, num_to_character)")
