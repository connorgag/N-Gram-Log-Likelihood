import pandas as pd
import math
import matplotlib.pyplot as plt 
pd.options.mode.chained_assignment = None


# ----------- Helper Functions ---------------
# Read in the vocab and unigram frequencies file and return a table with that information
def get_likelihoods(vocab_file, frequencies_file):
    # Read in vocab file
    vocab = []
    with open(vocab_file, "r") as file:
        for line in file:
            vocab.append(line.strip())

    # Read in frequencies file
    freq = []
    with open(frequencies_file, "r") as file:
        for line in file:
            freq.append(int(line.strip()))
    df = pd.DataFrame({"Word": vocab, "Occurrences": freq})
    df['Word'] = df['Word'].str.upper()
    return df

# Takes a sentence or word and turns it into the tokens in the vocab file
# Ignores punctuation for now
# Replaces unknown words with UNK and adds the starting token <s> 
def tokenize(vocab_file, sentence):
    # Read in vocab file
    vocab = []
    with open(vocab_file, "r") as file:
        for line in file:
            vocab.append(line.strip())

    # Always add the starting character
    tokenized_sentence = "<s> "

    # Ignoring punctuation - could change this later
    sentence = sentence.replace(".", "").replace(",", "").replace('"', "").upper()
    for word in sentence.split():
        if word in vocab:
            tokenized_sentence = tokenized_sentence + word + " "
        else:
            tokenized_sentence = tokenized_sentence + "<UNK> " 
    
    return tokenized_sentence


# ----------- Unigram ---------------
# Compute the unigram for all words in the vocab file and return a table with the probabilities
def ml_unigram_table(vocab_file, frequencies_file):
    likelihood_table = get_likelihoods(vocab_file, frequencies_file)
    total = likelihood_table["Occurrences"].sum()
    likelihood_table["Probability"] = likelihood_table["Occurrences"] / total
    return likelihood_table.sort_values(by=['Probability'], ascending=False)


# Compute the unigram for a word
# Assumes the word is in the vocab
def ml_unigram_word(vocab_file, frequencies_file, word):
    word = word.upper()

    table = ml_unigram_table(vocab_file, frequencies_file)
    # Check to see if the word is in the df
    # If it is, we can return that probability
    # If not, we need to add a small number to counteract this 0 probability
    # Since we've tokenized it this should never happen with the unigram
    if (table['Word'].str.contains(word).sum() == 0):
        return 1e-6
     
    return float(table[table['Word'] == word]['Probability'])


# Computes the log likelihood of a sentence using unigram
def log_likelihood_unigram(sentence, vocab_file, frequencies_file):
    sentence = tokenize(vocab_file, sentence)

    product = 1
    sentence_list = sentence.split()
    sentence_list.remove('<s>')

    print("Tokenized Sentence:")
    print(" ".join(sentence_list))
    for word in sentence_list:
        probability = ml_unigram_word(vocab_file, frequencies_file, word)
        product = product * probability

    return math.log(product)


# ----------- Bigram ---------------
# Creates a bigram table (word one is follow by word two, and this occurs this many times)
def get_bigram_table(vocab_file, bigram_file):
    # Read in vocab file
    vocab = []
    with open(vocab_file, "r") as file:
        for line in file:
            vocab.append(line.strip())

    with open(bigram_file, "r") as file:
        word_one_list = []
        word_two_list = []
        count_list = []
        for line in file:
            word_one, word_two, count = [int(i) for i in line.split()]
            word_one_list.append(vocab[word_one - 1])
            word_two_list.append(vocab[word_two - 1])
            count_list.append(count)

    df = pd.DataFrame({"Word_One": word_one_list, "Word_Two": word_two_list, "Count": count_list})
    df['Word_One'] = df['Word_One'].str.upper()
    df['Word_Two'] = df['Word_Two'].str.upper()

    return df

# Compute the words that are most likely to follow a certain word
# Returns a table of the likelihoods
def ml_bigram_table(vocab_file, bigram_file, word_to_follow):
    bigram_table = get_bigram_table(vocab_file, bigram_file)

    only_our_word = bigram_table[bigram_table['Word_One'] == word_to_follow.upper()]
    denominator = only_our_word['Count'].sum()
    only_our_word['Probability'] = only_our_word['Count'] / denominator
    return only_our_word.sort_values(by=['Probability'], ascending=False)[['Word_Two', 'Probability']]


# Compute the Bigram for a word
def ml_bigram_word(vocab_file, frequencies_file, word_one, word_two):
    word_one = word_one.upper()
    word_two = word_two.upper()

    table = ml_bigram_table(vocab_file, frequencies_file, word_one)
    # Check to see if the word is in the df
    # If it is, we can return that probability
    # If not, return a very small number because
    # If we return 0 then the whole probability will be 0
    if (table['Word_Two'].str.contains(word_two).sum() == 0):
        # print("Pair not observed: " + word_one + " " + word_two)
        return 1e-6

    return float(table[table['Word_Two'] == word_two]['Probability'])


# Computes the log likelihood of a sentence using bigram
def log_likelihood_bigram(sentence, vocab_file, frequencies_file):
    sentence = tokenize(vocab_file, sentence)
    print("Tokenized Sentence:")
    print(sentence)
    product = 1
    sentence_list = sentence.split()
    for i in range(1, len(sentence_list)):
        word_one = sentence_list[i-1]
        word_two = sentence_list[i]

        probability = ml_bigram_word(vocab_file, frequencies_file, word_one, word_two)
        product = product * probability       

    return math.log(product)



# ------------ Mixture Model ----------------

# Put two words in the mixture model
# Lambda value must be between 0 and 1 inclusive
# The higher that lambda value is, the more the unigram will take over and the less the bigram will take over
def mixture_model_by_word(vocab_file, unigram_file, bigram_file, word_one, word_two, lambda_value):
    unigram_of_word_two = ml_unigram_word(vocab_file, unigram_file, word_two)
    bigram_of_word_two_given_word_one = ml_bigram_word(vocab_file, bigram_file, word_one, word_two)

    result = (lambda_value * unigram_of_word_two) + ((1 - lambda_value) * bigram_of_word_two_given_word_one)
    return result


# Put a sentence in to the mixture model
def mixture_model_by_sentence(vocab_file, unigram_file, bigram_file, sentence, lambda_value):
    sentence = tokenize(vocab_file, sentence)

    product = 1
    sentence_list = sentence.split()
    for i in range(1, len(sentence_list)):
        word_one = sentence_list[i-1]
        word_two = sentence_list[i]
        probability = mixture_model_by_word(vocab_file, unigram_file, bigram_file, word_one, word_two, lambda_value)
        product = product * probability

    return math.log(product)


# Plot the mixture model to find the optimal value of lambda
# Doesn't return anything, just 
def plot_mixture(vocab_file, unigram_file, bigram_file, sentence):
    lambda_values = [i/100 for i in range(101)]
    mixture_values = []
    max_lambda= 0
    max_prob = float('-inf')
    for i in lambda_values:
        result = mixture_model_by_sentence(vocab_file, unigram_file, bigram_file, sentence, i)
        mixture_values.append(result)
        if (result > max_prob):
            max_prob = result
            max_lambda = i
        print(i)
    print("The max log probability is " + str(max_prob) + " with lambda " + str(max_lambda))

    plt.scatter(lambda_values, mixture_values)
    plt.title('Probability of Sentence: "' + sentence + '"')
    plt.xlabel("Lambda")
    plt.ylabel("Mixture Probability")
    plt.show()


def main():
    ## Question a
    # unigram_probs = ML_Unigram("hw4_vocab.txt", "hw4_unigram.txt")
    # print(unigram_probs[unigram_probs["Word"].str.startswith('M')])

    # Question b
    # bigram_probable_next_word_table = ML_Bigram_Table("hw4_vocab.txt", "hw4_bigram.txt", "THE")
    # print(bigram_probable_next_word_table.head(10))

    # Question c
    # sentence = "The stock market fell by one hundred points last week."
    # print('The unigram log likelihood of "' + sentence + '" is ' + str(log_likelihood_unigram(sentence, "hw4_vocab.txt", "hw4_unigram.txt")))
    # print('The bigram log likelihood of "' + sentence + '" is ' + str(log_likelihood_bigram(sentence, "hw4_vocab.txt", "hw4_bigram.txt")))

    # Question d
    # sentence = "The sixteen officials sold fire insurance."
    # print('The unigram log likelihood of "' + sentence + '" is ' + str(log_likelihood_unigram(sentence, "hw4_vocab.txt", "hw4_unigram.txt")))
    # print('The bigram log likelihood of "' + sentence + '" is ' + str(log_likelihood_bigram(sentence, "hw4_vocab.txt", "hw4_bigram.txt")))

    # Question e
    # Test for one lambda value
    # sentence = "The sixteen officials sold fire insurance."
    # lambda_value = 0
    # print(mixture_model_by_sentence("hw4_vocab.txt", "hw4_unigram.txt", "hw4_bigram.txt", sentence, lambda_value))

    # Print out the chart testing out all lambda values
    sentence = "The sixteen officials sold fire insurance."
    sentence = "The sixteen officials sold fire insurance."
    plot_mixture("hw4_vocab.txt", "hw4_unigram.txt", "hw4_bigram.txt", sentence)



main()