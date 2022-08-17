import argparse
import csv
import pathlib
import requests
import sys
import cologne_phonetics
import Levenshtein
import termcolor
import string
import spacy
from math import log10
from enum import Enum
from itertools import tee
from HanTa import HanoverTagger as ht
from difflib import SequenceMatcher
import collections

"""To run this script, go to terminal and execute: python main.py -i your_input.csv -o your_output.csv"""

# -----------------------------------------------------------------------------------------------
# PRESETS

# load Hannover Tagger for lemmatization/POS tagging  (needed for preprocessing & some scores)
HanTa_tagger = ht.HanoverTagger('morphmodel_ger.pgz')

# Define GENERAL_WORD_INFREQUENCY API endpoints (Score #1)
GENERAL_WORD_INFREQUENCY_API_DE = "https://www.dwds.de/api/frequency/?="

# Define WORD_COMBINATION_INFREQUENCY API endpoints (Score #2)
# TODO: To use the word_combination_infrequency score, replace `827FF4DDC28347C1A13FXXXXXXXXXXXX` with a valid API key from `https://www.scaleserp.com/`
# please also note, that the score is not inverted yet (so higher scores imply higher frequency aka less originality)
SCALESERP_API_KEY = "827FF4DDC28347C1A13FXXXXXXXXXXXX"    

def WORD_COMBINATION_INFREQUENCY_API_DE(q): # for GERMAN
    return "https://api.scaleserp.com/search?api_key=%s&q=%s&google_domain=google.de&location=Germany&gl=de&hl=de&page=1&output=json" % (
        SCALESERP_API_KEY, q)

# load SpaCy model for SEMANTIC_SIMILARITY score (Score #8)
# to install run " python -m spacy download de_core_news_lg "
nlp = spacy.load("de_core_news_lg") # large German model

# Define creativity scores
class Scores(Enum):
    GENERAL_WORD_INFREQUENCY = 1
    WORD_COMBINATION_INFREQUENCY = 2
    CONTEXT_SPECIFIC_WORD_UNIQUENESS = 3 
    SYNTAX_UNIQUENESS  = 4            
    RHYME = 5
    PHONETIC_SIMILARITY = 6
    SEQUENCE_SIMILARITY = 7 
    SEMANTIC_SIMILARITY = 8

# produces a colored output in the console
score_color_map = {
    Scores.GENERAL_WORD_INFREQUENCY: "yellow",
    Scores.WORD_COMBINATION_INFREQUENCY: "green",
    Scores.CONTEXT_SPECIFIC_WORD_UNIQUENESS: "red",
    Scores.SYNTAX_UNIQUENESS: "cyan",
    Scores.RHYME: "yellow",
    Scores.PHONETIC_SIMILARITY: "green",
    Scores.SEQUENCE_SIMILARITY: "red",
    Scores.SEMANTIC_SIMILARITY: "cyan"
}

#------------------------------------------------------------------------------------------------
## HELPER FUNCTIONS

def pairwise(iterable):
    f, s = tee(iterable)
    next(s, None)

    return zip(f, s)

# longest common subSEQUENCE (used in phonetic score)
def lcs(X, Y, m, n):
    if m == 0 or n == 0:
       return 0
    elif X[m-1] == Y[n-1]:
       return 1 + lcs(X, Y, m-1, n-1)
    else:
       return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n))

# -----------------------------------------------------------------------------------------------
# ARGUMENT PARSER
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="measure language creativity with phonetic analysis",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=120, width=99999))


    parser.add_argument("-i",
                        "--input_file",
                        metavar="INPUT_FILE",
                        help="the input file (e.gw., data/de_sentences.csv)",
                        action="store",
                        type=str,
                        required=True,
                        dest="input_file")
    parser.add_argument("-o",
                        "--output_file",
                        metavar="OUTPUT_FILE",
                        help="the output file (e.g., data/de_scores.csv)",
                        action="store",
                        type=str,
                        default=None,
                        dest="output_file")

    return parser.parse_args()

# ------------------------------------------------------------------------------------------------
## Creativity metrics / scores

# Score 1
def general_word_infrequency():
    score = 0
    frequencies = []

    # Set desired GENERAL_WORD_INFREQUENCY API (default is DE)
    gwi_api = GENERAL_WORD_INFREQUENCY_API_DE

    # Determine the novelty of each word in its lemmatized form
    for word in lemma_POS:
        lemma = word[1]
        freq_score = sys.maxsize
        q =  {'q': lemma}
        resp = requests.get(gwi_api, params=q)
        
        # Process only successful request responses
        if resp.status_code == 200:
            # The response from dwds.de/d/api contains the key `frequency`
            # which classifies words based on how frequently they are used in the corpus on a scale from 0-6
            freq_score = resp.json()["frequency"]
            freq_score = 6 - freq_score # invert scale
            score += freq_score

        if freq_score != sys.maxsize:
            frequencies.append(freq_score)

    #print(frequencies, score)

    # Calculate the arithmetic mean over all individual word infrequency scores
    return (score / 4)

# Score 2
def word_combination_infrequency():
    score = 0
    combination_info = {}

    # Set desired WORD_COMBINATION_INFREQUENCY API 
    combination_infrequency_api = WORD_COMBINATION_INFREQUENCY_API_DE

    # Iterate over all possible pairs of words in the sentence
    for first, second in pairwise(stripped_sentence):
        try:
            # Obtain the WORD_COMBINATION_INFREQUENCY from the scaleserp.com endpoint
            resp = requests.get(
                combination_infrequency_api("\"%s + %s\"" %
                                    (first, second)))

            # Process only successful request responses
            if resp.status_code == 200:
                response_json = resp.json()

                # Determine the frequency score in the nested JSON response
                # There are two preconditions for successful respones:
                # 1) response_json["request_info"]["success"] == True
                # 2) response_json["search_information"]["total_results"] != None
                if response_json["request_info"][
                        "success"] == True:
                    
                    if "total_results" in response_json[
                            "search_information"]:
                        # add info about results to dictionary
                        combination_info[response_json["search_information"][
                            "query_displayed"]] = response_json["search_information"]["total_results"] 
                    
                        # Map the number of google search results N on a log scale with base 10
                        score += log10(
                            int(response_json["search_information"]
                                ["total_results"]))        
                    else:
                        # If there are 0 search results, the word is highly novel
                        combination_info[response_json["search_information"][
                            "query_displayed"]] = "NA"
                        score += 0
        except Exception:
            continue

    # Perform the same calculation for word pairs that do not appear consecutively
    for indices in [[0, 2], [0, 3], [1, 3]]:
        try:
            resp = requests.get(
                combination_infrequency_api(
                    "\"%s * %s\"" %
                    (stripped_sentence[indices[0]],
                        stripped_sentence[indices[1]])))

            if resp.status_code == 200:
                response_json = resp.json()

                if response_json["request_info"][
                        "success"] == True:
                    if "total_results" in response_json[
                            "search_information"]:
                        
                        # add info about results to dictionary  
                        combination_info[response_json["search_information"][
                            "query_displayed"]] = response_json["search_information"]["total_results"]
                        # map score
                        score += log10(
                            int(response_json["search_information"]
                                ["total_results"]))
                    else:
                        combination_info[response_json["search_information"][
                            "query_displayed"]] = "NA"
                        score += 0
        except Exception:
            continue

    # Calculate the arithmetic mean over all individual word frequency scores
    return (score / 6) #NOTE: this score needs to be inverted manually during further analysis (in R)

# Score 3
def context_specific_word_uniqueness(min_variable_occurrence, max_variable_occurrence):
    score = 0
    # Score is calculated based on how often other participants have used the same word as a solution
    # to this specific four-letter word puzzle (variable)
    # These relative occurrences of words per variable are given in the variable_vocabulary dictionary
    
    lemma_sentence = lemma_samples[subject][variable]          
    lemma_stripped_sentence = lemma_sentence.strip(" ,;.:!?").lower().split() 

    # Get max & min occurrence for each variable (trial)
    max_variable_occurrence = variable_max_occurrence[variable]
    min_variable_occurrence = variable_min_occurrence[variable]

    # Substract min occurrance from each relative occurrence of each word in a given trial
    # average score over all 4 words of a sentence
    # compute final score by taking maximum occurrence across all sentences of a variable into account
    for word in lemma_stripped_sentence:
        score += variable_vocabulary[variable][word] - \
            min_variable_occurrence 

    score = (max_variable_occurrence - (score/4))/ \
        max_variable_occurrence 
    
    return score

# Score 4
def syntax_uniqueness(syntax_counts, min_syntax_count, max_syntax_count):
    score = 0

    # Eliminate special characters in the beginning or end
    stripped_sentence = sentence.strip(" ,;.:!?").split() 

    # Lemmatize and POS-tag the sentence using the Hannover Tagger (HanTa)
    lemma_POS = HanTa_tagger.tag_sent(stripped_sentence)
    
    # check syntax (get all 4 pos tags) of current sentence
    for i in range(4):
        sentence_pos_tags = []
        for word, tag in enumerate(lemma_POS):
            tag = lemma_POS[word][2]
            sentence_pos_tags.append(tag)
        syntax = " ".join(sentence_pos_tags)

    # compute score based on relative occurrence
    score = (max_syntax_count - (syntax_counts[syntax] - min_syntax_count)) / \
        max_syntax_count

    return score

# Score 5
def rhyme():
    score = 0

    phonetic_result = []
    # Eliminate all special characters on the left and right sides
    stripped_sentence = sentence.strip(" ,;.:!?").lower()
    
    # Compute the phonetic word representations
    # In German based on `cologne_phonetics`
    # `encode` allows for passing the entire sentence
    phonetic_result = cologne_phonetics.encode(
        stripped_sentence)
    # For English, you could use soundex instead:
    # In German based on `phonetics`, `soundex` accepts only one word per call
    # for word in stripped_sentence.split(" "):
    #    phonetic_result.append((word, phonetics.soundex(word)))

    sounds_to_word_groups = {}

    # Iterate over all phonetic word representations in pairs
    for i in range(len(phonetic_result)):
        for j in range(i + 1, len(phonetic_result)):
            (word1, sound1) = phonetic_result[i]
            (word2, sound2) = phonetic_result[j]

            # Iterate over all characters in the shorter phonetic word representation
            for x in range(1, 1 + min(len(sound1), len(sound2))):

                # Consider identical sounds as rhymes but ignore identical words
                if ((sound1[-x:] == sound2[-x:])
                        and not (word1 == word2)):

                    adj_word1 = word1
                    adj_word2 = word2

                    # Relevant for German: use d/t and s/z interchangeably
                    if x > 1:
                        if (adj_word1[-1:] == 'd'):
                            adj_word1 = adj_word1[:-1] + 't'
                        if (adj_word1[-1:] == 's'):
                            adj_word1 = adj_word1[:-1] + 'z'

                        if (adj_word2[-1:] == 'd'):
                            adj_word2 = adj_word2[:-1] + 't'
                        if (adj_word2[-1:] == 's'):
                            adj_word2 = adj_word2[:-1] + 'z'

                    # Go to the next word representation if the next considered sound differs
                    if not adj_word1[-(x + 1):] == adj_word2[-(
                            x + 1):]:
                        continue

                    # Initialize word group if the sound is encountered for the first time
                    if sound1[-x:] not in sounds_to_word_groups:
                        sounds_to_word_groups[sound1[-x:]] = set()

                    # Update the sound-to-word-group mapping
                    # sounds_to_word_groups maps a repeating sound to all the words that contain the sound
                    sounds_to_word_groups[sound1[-x:]].add(word1)
                    sounds_to_word_groups[sound1[-x:]].add(word2)
                else:
                    break

    word_groups_to_sounds = {}

    # Freeze the word groups to make them hash-able and therefore usable as keys in the
    # inverted map from word group --> sound
    for sound, word_group in sounds_to_word_groups.items():
        sounds_to_word_groups[sound] = frozenset(word_group)

    # Check if all values in sounds_to_word_groups are similar
    # If they all are similar, delete all, except for the first one  
    res = len(list(set(list(sounds_to_word_groups.values())))) == 1             
    #print("Are all values similar in dictionary? : " + str(res))
    if res == True:
        max_range = len(sounds_to_word_groups)
        for x in range(1,max_range):
            sounds_to_word_groups.popitem()

    # Build inverted map from word group --> sound
    # For every word group, choose the longest common sound
    for sound, word_group in sounds_to_word_groups.items():
        if (not word_group
                in word_groups_to_sounds) or len(sound) > len(
                    word_groups_to_sounds[word_group]):
            word_groups_to_sounds[word_group] = sound

    
    # Give score based on number of rhyming/similar word endings
    for word_group, sound in word_groups_to_sounds.items():
        rhyme_length = len(sound)
        num_words_in_rhyme = len(word_group)
        if num_words_in_rhyme == 4:
            score = 4
        else:    
            score = (num_words_in_rhyme * rhyme_length)-1

    return score

# Score 6
def phonetic_similarity():
    score = 0

    total_combinations = 0
    levenstein_score = 0
    substring_score = 0
    
    phonetic_result = []
    
    # Eliminate all special characters on the left and right sides
    stripped_sentence = sentence.strip(" ,;.:!?").lower()

    # for German, see rhyme score for English alternative suggestion
    phonetic_result = cologne_phonetics.encode(
        stripped_sentence)

    # Iterate over the phonetic word representations in pairs
    for i in range(len(phonetic_result)):
        for j in range(i + 1, len(phonetic_result)):
            total_combinations += 1

            (word1, sound1) = phonetic_result[i]
            (word2, sound2) = phonetic_result[j]

            # Compute the Levenshtein distance between the two representations
            levenstein_distance = Levenshtein.distance(
                sound1, sound2)
            # Compute the longest substring between the two representations
            longest_substr_len = lcs(sound1, sound2, len(sound1), len(sound2))

            # Normalize the Levenshtein distance score using the lenght of the longer representation
            levenstein_score += levenstein_distance / max(
                len(sound1), len(sound2))

            # Normalize the longest substring score using the lenght of the shorter representation
            substring_score += longest_substr_len / min(
                len(sound1), len(sound2))

    # Map both partial phonetic scores to a scale from 0 to 1
    score += 0.5 * (1 -
                        (levenstein_score / total_combinations))
    score += 0.5 * (substring_score / total_combinations)

    return score

# Score 7
def sequence_similarity():
    score = 0
    ratios = []

    stripped_sentence = sentence.strip(" ,;.:!?").upper().split()
    
    # determine similarity ratio using difflib / Gestalt pattern matching
    for word in stripped_sentence:                 
        ratio = SequenceMatcher(None, variable, word).ratio()
        ratios.append(ratio)

    score = max(ratios)

    return score

# Score 8
def semantic_similarity():
    score = 0

    cap_variable = variable.capitalize()
    variable_doc = nlp(cap_variable) # spacy model needs to be loaded for this

    lemmas = []
    # get lemmas from HanTa (lemmatization worked better than spacy)
    for lemma in lemma_POS:
        word = lemma[1]
        lemmas.append(word)
    
    # convert list to string, create doc
    lemmas_doc = nlp(' '.join(lemmas))
    
    sem_similarity_results = [] 

    if variable_doc.vector_norm != 0:
        for word in lemmas_doc:
            if word.vector_norm == 0:
                score = 99*4
                break # jump out of loop
            else:
                #print(variable_doc, "<->", word, variable_doc.similarity(word))
                score += abs(variable_doc.similarity(word))
                sem_similarity_results.append(variable_doc.similarity(word))
    else:
        score = 99*4 
        
    # add score to class
    return (score/4)

#-------------------------------------------------------------------------------------------------
## MAIN 
if __name__ == "__main__":
    arguments = parse_arguments()

    # Read input file containing subjects (anonymous participant id), variables (given four letter word),
    # and sentences (participants "creative" solution)   
    data = csv.reader(open(arguments.input_file, "r"), delimiter=";") 
    next(data)

    samples = {}
    lemma_samples = {}
    scores = {}

    all_pos_tags = [] # for syntax score

    # PREPROCESSING STEPs
    for line in data:
        stripped_sentence = "".join([i for i in line[2] if i not in string.punctuation]).upper().split()

        # 1. Discard sentences that are not of length 4 (meaning sentences that do not contain exactly 4 words)
        if len(stripped_sentence) != 4:
            continue

        # 2. Discard sentences that contain words which do not start with correct letter
        # according to given variable (i.e. EKEL)
        for word in stripped_sentence:
            if word[0] not in line[1]:
                continue
        
        # 3. Lemmatization and part-of-speech tagging of sentences
        stripped_sentence_2 = "".join([i for i in line[2] if i not in string.punctuation]).split() # without lower()
        lemma_POS2 = HanTa_tagger.tag_sent(stripped_sentence_2)

        # extract word lemmas and pos tags separately
        lemmas = []
        for lemma in lemma_POS2:
            word = lemma[1]
            lemmas.append(word)

        lemma = [' '.join(lemmas)]

        for i in range(4):
            sentence_pos_tags = []
            for word, tag in enumerate(lemma_POS2):
                tag = lemma_POS2[word][2]
                sentence_pos_tags.append(tag)
            sentence_pos_tags = " ".join(sentence_pos_tags)

        all_pos_tags.append(sentence_pos_tags)
        
    	# 4. Build data structure with dictionaries
        if line[0] not in samples:
            samples[line[0]] = {}
            lemma_samples[line[0]] = {}
            scores[line[0]] = {}

        samples[line[0]][line[1]] = line[2]
        lemma_samples[line[0]][line[1]] = lemma[0]
        scores[line[0]][line[1]] = {}
    
    # Preparation for context-specific solution score (uniqueness of word choice in per trial) -----------
    subject_vocabulary = {}
    variable_vocabulary = {} # vocab per trial
    variable_word_count = {} # number of words per trial

    for subject, pairs in lemma_samples.items(): # uses lemmas
        if subject not in subject_vocabulary:
            subject_vocabulary[subject] = {}

        for variable, sentence in pairs.items():

            # Count number of words per trial (i.e., cue)
            if variable not in variable_word_count:
                variable_word_count[variable] = 4 # each sentence has 4 words
            else:
                variable_word_count[variable] += 4

            if variable not in variable_vocabulary:
                variable_vocabulary[variable] = {}

            for word in sentence.strip(" ,;.:!?").lower().split():
                if word not in subject_vocabulary[subject]:
                    subject_vocabulary[subject][word] = 1
                else:
                    subject_vocabulary[subject][word] += 1

                if word not in variable_vocabulary[variable]:
                    variable_vocabulary[variable][word] = 1
                else:
                    variable_vocabulary[variable][word] += 1

        # Calculate relative occurrences of words per variable (i.e., cue) or subject
    for subject, pairs in subject_vocabulary.items():
        for word, count in pairs.items():
            subject_vocabulary[subject][word] /= len(
                subject_vocabulary[subject])

    for variable, pairs in variable_vocabulary.items():
        for word, count in pairs.items():
            variable_vocabulary[variable][word] /= \
                variable_word_count[variable]

    max_subject_occurrence = 0.0
    min_subject_occurrence = 1.0
    for subject in subject_vocabulary.values():

        if max(subject.values()) > max_subject_occurrence:
            max_subject_occurrence = max(subject.values())

        if min(subject.values()) < min_subject_occurrence:
            min_subject_occurrence = min(subject.values())

        # Create new dict for each variables (trial) max and min occurrence
    variable_max_occurrence = {}
    variable_min_occurrence = {}
    
    for variable, occurrence in variable_vocabulary.items():
        max_variable_occurrence = 0.0
        min_variable_occurrence = 1.0

        if max(occurrence.values()) > max_variable_occurrence:
            max_variable_occurrence = max(occurrence.values())

        if min(occurrence.values()) < min_variable_occurrence:
            min_variable_occurrence = min(occurrence.values()) 

        variable_max_occurrence[variable] = max_variable_occurrence
        variable_min_occurrence[variable] = min_variable_occurrence

    # Preparation for syntax uniqueness score: determine relative occurrence of a combination of POS tags --------
    syntax_counts = dict(collections.Counter(all_pos_tags))

    s = sum(syntax_counts.values())
    for syntax, count in syntax_counts.items():
        syntax_counts[syntax] /= s

    min_syntax_count = min(syntax_counts.values())
    max_syntax_count = max(syntax_counts.values())

    # COMPUTE CREATIVITY SCORES (text metrics) for all subjects and their variables
    for subject_index, (subject, pairs) in enumerate(samples.items()):

        for variable_index, (variable, sentence) in enumerate(pairs.items()):
            
            # Eliminate special characters in the beginning or end
            stripped_sentence = sentence.strip(" ,;.:!?").lower().split()

            # Lemmatize and POS-tag the sentence using the Hannover Tagger (HanTa)
            lemma_POS = HanTa_tagger.tag_sent(stripped_sentence)
            #print (lemma_POS)

            # Compute scores
            scores[subject][variable][Scores.GENERAL_WORD_INFREQUENCY] = general_word_infrequency() # 1
            scores[subject][variable][Scores.WORD_COMBINATION_INFREQUENCY] = word_combination_infrequency() #2
            scores[subject][variable][Scores.CONTEXT_SPECIFIC_WORD_UNIQUENESS] = context_specific_word_uniqueness(min_variable_occurrence, max_variable_occurrence) #3
            scores[subject][variable][Scores.SYNTAX_UNIQUENESS] = syntax_uniqueness(syntax_counts, min_syntax_count, max_syntax_count) #4
            scores[subject][variable][Scores.RHYME] = rhyme() #5
            scores[subject][variable][Scores.PHONETIC_SIMILARITY] = phonetic_similarity() #6
            scores[subject][variable][Scores.SEQUENCE_SIMILARITY] = sequence_similarity() #7
            scores[subject][variable][Scores.SEMANTIC_SIMILARITY] = semantic_similarity() #8

            # CSV Output -------------------------------------------------------------------------
            # Construct the CSV header rows
            plain_header = ""
            colored_header = ""

            # Print the header line in the first iteration, displays the title of each column for the following lines
            if subject_index == 0 and variable_index == 0:
                plain_header = "\"subject\",\"variable\",\"sentence\""
                colored_header = plain_header

                for score in Scores:
                    plain_header += ",\"%s\"" % (score.name)
                    colored_header += ",%s" % (termcolor.colored(
                        "\"%s\"" % (score.name), score_color_map[score]))
                
                plain_header += "\n"
                colored_header += "\n"

            # Print the score line (in color for the console output)
            plain_output = plain_header + "\"%s\",\"%s\",\"%s\"" % (
                subject, variable, sentence)
            colored_output = colored_header + "\"%s\",\"%s\",\"%s\"" % (
                subject, variable, sentence)

            for score in Scores:
                plain_output += ",%s" % (scores[subject][variable][score])
                colored_output += ",%s" % (termcolor.colored(
                    scores[subject][variable][score],
                    score_color_map[score]))

            print(colored_output)

            # Optionally, write the results to the output file
            if arguments.output_file:
                output_file = pathlib.Path(arguments.output_file).resolve()

                with output_file.open("a") as f:
                    f.write(plain_output + "\n")