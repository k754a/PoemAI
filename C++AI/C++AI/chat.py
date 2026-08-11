import os
import numpy as np

# Optional rhyming. Install with: pip install pronouncing
try:
    import pronouncing
    USE_RHYMES = True
except Exception:
    USE_RHYMES = False

# --- LOAD MODEL SAFELY ---
print("Loading model...")

if not os.path.exists("poem_model.txt"):
    print("ERROR: poem_model.txt not found. Train the C++ model first!")
    exit()

with open("poem_model.txt", encoding="utf-8", errors="ignore") as f:
    word_count = int(f.readline())
    neuron_count = int(f.readline())
    vocab_size = int(f.readline())

    id2word, word2id = {}, {}
    for _ in range(vocab_size):
        line = f.readline().strip()
        if not line: 
            continue
        parts = line.split(" ", 1)
        if len(parts) < 2: 
            continue
        
        wid, word = int(parts[0]), parts[1]
        id2word[wid] = word
        word2id[word] = wid

    def safe_load(rows):
        data = []
        for _ in range(rows):
            vals = []
            for x in f.readline().split():
                try:
                    v = float(x)
                    vals.append(v if np.isfinite(v) else 0.0)
                except ValueError:
                    vals.append(0.0)
            
            # Pad or trim to ensure exact shape match
            while len(vals) < neuron_count:
                vals.append(0.0)
            vals = vals[:neuron_count]
            data.append(vals)
            
        return np.array(data, dtype=np.float32)

    weights = safe_load(word_count)
    out_weights = safe_load(word_count)

print(f"Model loaded: {vocab_size} words, {neuron_count} neurons")

# Low IDs are common words (tokenizer sorts by frequency).
# Used as fallback for unknown prompts.
common_ids = [i for i in range(1, min(500, word_count)) if id2word.get(i, "") != "<NEWLINE>"]
if not common_ids:
    common_ids = list(range(1, min(10, word_count)))

# --- CHANGED: Match your new C++ context size! ---
CONTEXT = 12 
MAX_TOKENS = 90
MAX_LINES = 4

print("\n--- AI Poet (type 'exit' to stop) ---")

while True:
    text = input("> ").strip().lower()
    if text == "exit": 
        break
    if not text: 
        continue

    # Convert prompt to IDs, filter out unknowns
    ids = [word2id.get(w, 0) for w in text.split()]
    ids = [i for i in ids if i > 0]

    # Fallback: If prompt is entirely unknown, seed with a common word
    if not ids:
        ids = [int(np.random.choice(common_ids))]

    generated_words = []
    last_line_end_word = ""

    for step in range(MAX_TOKENS):
        # Stop after generating enough lines
        if generated_words.count("<NEWLINE>") >= MAX_LINES:
            break

        # Use recent tokens as context
        current_ids = ids[-CONTEXT:]

        # 1. MATH MATCH: Recency Bias + Average + Clamp (Matches your new C++ code exactly!)
        hidden = np.zeros(neuron_count, dtype=np.float32)
        L = len(current_ids)
        
        for i, wid in enumerate(current_ids):
            if wid > 0:
                # Map index so the newest word always gets the highest weight (w = CONTEXT - 1)
                # This perfectly mirrors your C++ `float importance = (float)(w + 1) / context;`
                w = i + (CONTEXT - L)
                importance = (w + 1) / float(CONTEXT)
                hidden += weights[wid] * importance
        
        # Divide by context and clamp, exactly like C++
        hidden /= float(CONTEXT)
        hidden = np.clip(hidden, -1.0, 1.0)

        # Score all words
        scores = out_weights @ hidden
        scores = scores.astype(np.float32)
        scores[0] = -999999.0  # Ban padding ID

        # Tie-breaker: Slight bias toward lower IDs (common words)
        scores[1:] -= np.arange(1, word_count, dtype=np.float32) * 0.000001
        
        # Tiny noise for variety
        scores += np.random.normal(0.0, 0.001, size=scores.shape).astype(np.float32)

        # 2. REPETITION PENALTY: Ban recent non-newline words
        recent_words = [w for w in generated_words[-20:] if w != "<NEWLINE>"][-12:]
        for used_word in recent_words:
            used_id = word2id.get(used_word, 0)
            if used_id > 0:
                scores[used_id] = -999999.0

        # Count words since last newline
        words_since_newline = 0
        for w in reversed(generated_words):
            if w == "<NEWLINE>": 
                break
            words_since_newline += 1

        # 3. NEWLINE CONTROL: Force poem-like structure
        newline_id = word2id.get("<NEWLINE>", 0)
        if newline_id > 0:
            if len(generated_words) == 0 or words_since_newline < 4:
                scores[newline_id] = -999999.0
            elif words_since_newline >= 7:
                # Heavily encourage line break after 7+ words
                scores[newline_id] += 5.0

        # Get top valid candidates
        top_indices = np.argsort(scores)[-25:][::-1]
        top_indices = [int(i) for i in top_indices if i > 0 and scores[i] > -999998.0]

        if not top_indices:
            best_id = int(np.random.choice(common_ids))
        else:
            best_id = top_indices[0]

            # 4. RHYME TRICK: Override with rhyme if available
            if USE_RHYMES and last_line_end_word and words_since_newline >= 2:
                try:
                    rhymes = pronouncing.rhymes(last_line_end_word.lower())
                except Exception:
                    rhymes = []
                
                found_rhyme = False
                for idx in top_indices[:25]:
                    cand = id2word.get(idx, "")
                    if cand and cand.lower() in rhymes:
                        best_id = idx
                        found_rhyme = True
                        break
                
                # No rhyme found? Pick randomly from top 3 for variety
                if not found_rhyme:
                    best_id = int(np.random.choice(top_indices[:min(3, len(top_indices))]))
            else:
                # Normal generation: Pick randomly from top 3
                best_id = int(np.random.choice(top_indices[:min(3, len(top_indices))]))

        next_word = id2word.get(best_id, "?")

        # Track last word before newline for future rhymes
        if next_word == "<NEWLINE>":
            if generated_words and generated_words[-1] != "<NEWLINE>":
                last_line_end_word = generated_words[-1]

        generated_words.append(next_word)
        ids.append(best_id)

    # Format and print poem
    poem = " ".join(generated_words)
    poem = poem.replace(" <NEWLINE> ", "\n").replace("<NEWLINE>", "\n").strip()
    
    print("\nAI Poem:")
    print(poem)
    print()