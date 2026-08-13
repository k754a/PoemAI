
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
common_ids = list(range(1, min(500, word_count)))

if not common_ids:
    common_ids = list(range(1, min(10, word_count)))

CONTEXT = 12
MAX_TOKENS = 120

print("\n--- AI Poet (type 'exit' to stop) ---")

while True:
    text = input("> ").strip().lower()

    if text == "exit":
        break

    if not text:
        continue

    # Convert prompt to IDs
    prompt_words = text.split()

    ids = [
        word2id.get(w, 0)
        for w in prompt_words
    ]

    ids = [
        i for i in ids
        if i > 0
    ]

    # Use the first input word as the starting word.
    # This lets the model build the poem from the user's word.
    first_word = prompt_words[0]
    first_id = word2id.get(first_word, 0)

    # Fallback if the first word isn't in the vocabulary
    if first_id <= 0:
        first_id = int(np.random.choice(common_ids))
        first_word = id2word.get(first_id, "?")

    # Start generation with the user's first word
    generated_words = [first_word]
    ids = [first_id]

    # Keep track of every generated word.
    # A word can only appear once in the poem.
    used_ids = {first_id}

    for step in range(MAX_TOKENS - 1):

        # Use recent tokens as context
        current_ids = ids[-CONTEXT:]

        # ---------------------------------------------------------
        # 1. MATH MATCH
        # Matches the C++ code:
        #
        # hiddenState[n] += weights[wordId][n]
        # hiddenState[n] /= context
        # clamp to [-1, 1]
        # ---------------------------------------------------------

        hidden = np.zeros(
            neuron_count,
            dtype=np.float32
        )

        count = 0

        for wid in current_ids:
            if wid > 0:
                hidden += weights[wid]
                count += 1

        # Divide by CONTEXT to match C++ exactly
        if count > 0:
            hidden /= float(CONTEXT)
            hidden = np.clip(
                hidden,
                -1.0,
                1.0
            )

        # ---------------------------------------------------------
        # 2. SCORE ALL WORDS
        # ---------------------------------------------------------

        scores = out_weights @ hidden
        scores = scores.astype(np.float32)

        # Ban padding ID
        scores[0] = -999999.0

        # Tiny bias toward common words
        scores[1:] -= (
            np.arange(
                1,
                word_count,
                dtype=np.float32
            ) * 0.000001
        )

        # Tiny noise for variety
        scores += np.random.normal(
            0.0,
            0.001,
            size=scores.shape
        ).astype(np.float32)

        # ---------------------------------------------------------
        # 3. NEVER REPEAT A WORD
        # ---------------------------------------------------------

        for used_id in used_ids:
            if used_id > 0 and used_id < word_count:
                scores[used_id] = -999999.0

        # ---------------------------------------------------------
        # 4. GET BEST CANDIDATES
        # ---------------------------------------------------------

        top_indices = np.argsort(scores)[-25:][::-1]

        top_indices = [
            int(i)
            for i in top_indices
            if (
                i > 0
                and scores[i] > -999998.0
                and i not in used_ids
            )
        ]

        # ---------------------------------------------------------
        # 5. CHOOSE NEXT WORD
        # ---------------------------------------------------------

        if not top_indices:
            break

        # Pick randomly from top 3 to keep some variety
        best_id = int(
            np.random.choice(
                top_indices[
                    :min(3, len(top_indices))
                ]
            )
        )

        next_word = id2word.get(
            best_id,
            "?"
        )

        # If somehow invalid, stop
        if next_word == "?":
            break

        # Add the new word
        generated_words.append(next_word)
        ids.append(best_id)
        used_ids.add(best_id)

    # -------------------------------------------------------------
    # PRINT POEM
    # -------------------------------------------------------------

    poem = " ".join(generated_words)

    print("\nAI Poem:")
    print(poem)
    print()
