"""
 * The MIT License
 *
 * Copyright 2021 The OpenNARS authors.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * """

"""
English input channel for OpenNARS for Applications.
A shallow semantic parser with basic grammar learning ability
by using NAL REPRESENT relations.

Updated for modularity, to be called within other scripts
"""

import re
import sys
import time
import subprocess
import nltk as nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet

# Global variables
SyntacticalTransformations = [
    #types of tuples of words with optional members
    (r" VERB_([0-9]*) VERB_([0-9]*) ", r" VERB_\1 ADJ_\2 "), #hack for the lousy nltk postagger (verbs don't come in succession, DET would have been detected, ADJ is better guess)
    (r" BE_([0-9]*) ADP_([0-9]*) ", r" ADP_\2 "), #(optional learnable)
    (r" BE_([0-9]*) ADV_VERB_([0-9]*) ", r" ADV_VERB_\2 "), #(optional learnable)
    (r" DET_([0-9]*) ", r" "), #ignore determiner
    (r" ADJ_([0-9]*) NOUN_([0-9]*) ", r" ADJ_NOUN_\2 "),
    (r" NOUN_([0-9]*) ", r" ADJ_NOUN_\1 "),
    (r" ADV_([0-9]*) VERB_([0-9]*) ", r" ADV_VERB_\2 "),
    (r" VERB_([0-9]*) ", r" ADV_VERB_\1 "),
]

TermRepresentRelations = [
    #subject, predicate, object encoding
    (r"ADJ_NOUN_([0-9]*)", "( [ %s ] & %s )", (1.0, 0.99)),
    (r"ADV_VERB_([0-9]*)", "( [ %s ] & %s )", (1.0, 0.99))
]

StatementRepresentRelations = [
    #clauses to Narsese:
    (r"\A(.*) IF_([0-9]*) (.*)\Z", r" < \3 =/> \1 > ", (1.0, 0.99), 0), #Conditional
    (r" ADJ_NOUN_([0-9]*) ADV_VERB_([0-9]*) ADP_([0-9]*) ADJ_NOUN_([0-9]*) ", r" < ( ADJ_NOUN_\1 * ADJ_NOUN_\4 ) --> ADV_VERB_\2+ADP_\3 > ", (1.0, 0.99), 0), #new addition for lie_in above_of etc.
    (r" ADJ_NOUN_([0-9]*) BE_([0-9]*) ADP_([0-9]*) ADJ_NOUN_([0-9]*) ", r" < ( ADJ_NOUN_\1 * ADJ_NOUN_\4 ) --> BE_\2+ADP_\3 > ", (1.0, 0.99), 0), #new addition for lie_in above_of etc.
    (r" ADJ_NOUN_([0-9]*) BE_([0-9]*) ADJ_NOUN_([0-9]*) ADP_([0-9]*) ADJ_NOUN_([0-9]*) ", r" < ( ADJ_NOUN_\1 * ADJ_NOUN_\5 ) --> ADJ_NOUN_\3+ADP_\4 > ", (1.0, 0.99), 0), #new addition for lie_in above_of etc.
    (r" ADJ_NOUN_([0-9]*) BE_([0-9]*) ADJ_([0-9]*) ADP_([0-9]*) ADJ_NOUN_([0-9]*) ", r" < ( ADJ_NOUN_\1 * ADJ_NOUN_\5 ) --> ADJ_\3+ADP_\4 > ", (1.0, 0.99), 0), #new addition for larger_than etc.
    (r" ADJ_NOUN_([0-9]*) ADV_VERB_([0-9]*) ADJ_NOUN_([0-9]*) ADJ_NOUN_([0-9]*) ", r" <(( ADJ_NOUN_\1 * ADJ_NOUN_\3 ) * ADJ_NOUN_\4 ) --> ADV_VERB_\2 > ", (1.0, 0.99), 0), #SVOO
    (r" ADJ_NOUN_([0-9]*) BE_([0-9]*) ADJ_NOUN_([0-9]*) ", r" < ADJ_NOUN_\1 --> ADJ_NOUN_\3 > ", (1.0, 0.99), 0), #SVC
    (r" ADJ_NOUN_([0-9]*) ADV_VERB_([0-9]*) ADJ_NOUN_([0-9]*) ", r" <( ADJ_NOUN_\1 * ADJ_NOUN_\3 ) --> ADV_VERB_\2 > ", (1.0, 0.99), 0), #SVO
    (r" ADJ_NOUN_([0-9]*) BE_([0-9]*) ADJ_([0-9]*) ", r" < ADJ_NOUN_\1 --> [ ADJ_\3 ]> ", (1.0, 0.99), 0), #SVC
    (r" ADJ_NOUN_([0-9]*) ADP_([0-9]*) ADJ_NOUN_([0-9]*) ", r" <( ADJ_NOUN_\1 * ADJ_NOUN_\3 ) --> ADP_\2 > ", (1.0, 0.99), 0), #S*A (part1)
    (r" ADJ_NOUN_([0-9]*) (.*) ADP_([0-9]*) ADJ_NOUN_([0-9]*) ", r" ADJ_NOUN_\1 \2 , < ( ADJ_NOUN_\1 * ADJ_NOUN_\4 ) --> ADP_\3 > ", (1.0, 0.90), 0), #S*A (part2, optional learnable)
    (r" ADJ_NOUN_([0-9]*) ADV_VERB_([0-9]*) ", r" < ADJ_NOUN_\1 --> [ ADV_VERB_\2 ] > ", (1.0, 0.99), 0), #SV
]

class EnglishToNarsese:
    def __init__(self, verbose=False, output_truth=False, eternal_output=False, nltk_data_path=None):
        self.verbose = verbose
        self.output_truth = output_truth
        self.eternal = eternal_output
        self.tense_from_sentence = True
        self.current_time = 0
        self.motivation = None
        self.thinkcycles = None
        self.acquired_grammar = []
        
        quiet = True;
        # * This only needs to be done once per machine, afterwards it can stay commented out
        # * Download required NLTK data
        # if nltk_data_path:
        #     nltk.data.path.append(nltk_data_path)
        
        # quiet = not verbose
        nltk.download('punkt', quiet=quiet)
        nltk.download('averaged_perceptron_tagger', quiet=quiet)
        nltk.download('universal_tagset', quiet=quiet)
        nltk.download('wordnet', quiet=quiet)
        nltk.download('omw-1.4', quiet=quiet)
        nltk.download('punkt_tab', quiet=quiet)
        nltk.download('averaged_perceptron_tagger_eng', quiet=quiet)
    
    # Convert universal tag set to the wordnet word types
    def wordnet_tag(self, tag):
        if tag == "ADJ":
            return wordnet.ADJ
        elif tag == "VERB":
            return wordnet.VERB
        elif tag == "NOUN":
            return wordnet.NOUN
        elif tag == 'ADV':
            return wordnet.ADV
        else:          
            return wordnet.NOUN  # default
    
    # POS-tag words in the input sentence and lemmatize them using Wordnet
    def sentence_and_types(self, text):
        tokens = [word for word in word_tokenize(text)]
        wordtypes_ordered = nltk.pos_tag(tokens, tagset='universal')
        wordtypes = dict(wordtypes_ordered)
        lemma = WordNetLemmatizer()
        handleInstance = lambda word: "{"+word+"}" if word[0].isupper() else word
        tokens = [handleInstance(lemma.lemmatize(word, pos=self.wordnet_tag(wordtypes[word]))) for word in tokens]
        wordtypes = dict([(tokens[i], wordtypes_ordered[i][1]) for i in range(len(tokens))])
        wordtypes = {key: ("BE" if key == "be" else ("IF" if key == "if" else ("NOUN" if value=="PRON" or value=="NUM" else ("ADP" if value=="PRT" else value)))) 
                    for (key, value) in wordtypes.items()}
        indexed_wordtypes = []
        i = 0
        lasttoken = None
        for token in tokens:
            if lasttoken == None or wordtypes[lasttoken] == "NOUN" or wordtypes[token] == "ADP" or wordtypes[token] == "IF":  # adjectives don't cross these
                i += 1  # each noun or new article ends previous ADJ_NOUN index
            indexed_wordtypes.append(wordtypes[token] + "_" + str(i))
            lasttoken = token
        if self.verbose:
            print("//Word types: " + str(wordtypes))
        return " " + " ".join(tokens) + " ", " " + " ".join(indexed_wordtypes) + " "

    # NAL truth functions
    def truth_deduction(self, Ta, Tb):
        return [Ta[0]*Tb[0], Ta[0]*Tb[0]*Ta[1]*Tb[1]]

    def truth_w2c(self, w):
        return w / (w + 1.0)

    def truth_c2w(self, c):
        return c / (1.0 - c)

    def truth_expectation(self, v):
        return (v[1] * (v[0] - 0.5) + 0.5)

    def truth_revision(self, v1, v2):
        (f1, c1) = v1
        (f2, c2) = v2
        w1 = self.truth_c2w(c1)
        w2 = self.truth_c2w(c2)
        w = w1 + w2
        return (min(1.0, (w1 * f1 + w2 * f2) / w), 
                min(0.99, max(max(self.truth_w2c(w), c1), c2)))

    # Return the concrete word (compound) term
    def get_word_term(self, term, cur_truth, suppress_output=True):
        for (schema, compound, Truth) in TermRepresentRelations:
            m = re.match(schema, term)
            if not m:
                continue
            cur_truth[:] = self.truth_deduction(cur_truth, Truth)
            modifier = term.split("_")[0] + "_" + m.group(1)
            atomic = term.split("_")[1] + "_" + m.group(1)
            if modifier in self.word_type:
                if self.verbose and not suppress_output:
                    print("// Using " + str((schema, compound, Truth)))
                term = compound % (self.word_type[modifier], self.word_type[atomic]) 
            else:
                term = atomic
        return self.word_type.get(term, term)

    # Apply syntactical reductions and wanted represent relations
    def reduce_typetext(self, typetext, apply_statement_represent=False, apply_term_represent=False, suppress_output=True):
        cur_truth = [1.0, 0.9]
        for i in range(len(SyntacticalTransformations)):
            for (a, b) in SyntacticalTransformations:
                typetext = re.sub(a, b, typetext)
        if apply_statement_represent:
            for (a, b, Truth, _) in self.acquired_grammar + StatementRepresentRelations:
                typetext_new = re.sub(a, b, typetext)
                if typetext_new != typetext:
                    if self.verbose and not suppress_output:
                        print("// Using " + str((a, b, Truth)))
                    typetext = typetext_new
                    cur_truth = self.truth_deduction(cur_truth, Truth)
            if apply_term_represent:
                typetext = " ".join([self.get_word_term(x, cur_truth, suppress_output=suppress_output) 
                                   if "+" not in x else 
                                   self.get_word_term(x.split("+")[0], cur_truth, suppress_output=suppress_output)+"_"+
                                   self.get_word_term(x.split("+")[1], cur_truth, suppress_output=suppress_output) 
                                   for x in typetext.split(" ")])
        return typetext, cur_truth

    # Learn grammar pattern
    def grammar_learning(self, y="", forced=False):
        if forced or (not y.startswith("<") or not y.endswith(">") or (y.count("<") > 1 and not "=/>" in y)):  # Only if not fully encoded/valid Narsese
            print("//What? Tell \"" + self.sentence.strip() + "\" in simple sentences: (newline-separated)")
            L = []
            while True:
                try:
                    s = " " + input().rstrip("\n") + " "
                    print("//Example input: " + s.strip() if s.strip() != "" else "//Example done.")
                except:
                    return False
                if s.strip() == "":
                    break
                L.append(self.sentence_and_types(s)[0])
            mapped = ",".join([self.reduce_typetext(" " + " ".join([self.type_word.get(x) for x in part.split(" ") 
                                                                 if x.strip() != "" and x in self.type_word]) + " ")[0] 
                              for part in L])
            if mapped.strip() != "":
                (R, mapped, T) = (self.reduce_typetext(self.typetext_reduced)[0], mapped, (1.0, 0.45))
                for i, typeword in enumerate(R.strip().split(" ")):  # generalize grammar indices
                    R = R.replace(typeword, "_".join(typeword.split("_")[:-1]) + "_([0-9]*)")
                    mapped = mapped.replace(typeword, "_".join(typeword.split("_")[:-1])+"_\\" + str(i+1))
                for (R2, mapped2, T2, _) in self.acquired_grammar:
                    if R == R2 and mapped == mapped2:
                        T = self.truth_revision(T, T2)
                        break
                print("//Induced grammar relation: " + str((R, mapped, T)))
                sys.stdout.flush()
                self.current_time += 1
                self.acquired_grammar.append((R, mapped, T, self.current_time))
                self.acquired_grammar.sort(key=lambda T: (-self.truth_expectation(T[2]), -T[3]))
            return True
        return False

    def process_line(self, line):
        """Process a single input line and return Narsese output"""
        self.current_time += 1
        
        if len(line) == 0:
            return "\n"
            
        is_question = line.endswith("?")
        is_goal = line.endswith("!")
        is_command = line.startswith("*") or line.startswith("//") or line.isdigit() or line.startswith('(') or line.startswith('<') or line.endswith(":|:")
        spaced_line = (" " + line.lower() + " ")
        is_negated = " not " in spaced_line or " no " in spaced_line
        
        # Handle commands
        if is_command:
            if line.startswith("*eternal=false"):
                self.eternal = False
                return ""
            if line.startswith("*eternal=true"):
                self.eternal = True
                return ""
            if line.startswith("*motivation="):
                self.motivation = line.split("*motivation=")[1]
                return ""
            if line.startswith("*thinkcycles="):
                self.thinkcycles = line.split("*thinkcycles=")[1]
                return ""
            if line.startswith("*teach"):
                self.grammar_learning(forced=True)
                return ""
            else:
                return line
        
        results = []
        if line.strip() != "":
            # results.append("//Input sentence: " + line)
            results.append("")
        
        # Determine tense from sentence
        punctuations = [" ", "!", "?"]
        tenses_past = ["previously", "before"]
        tenses_present = ["now", "currently", "afterwards"]
        tenses_future = ["afterwards", "later"]
        event_tenses = tenses_past + tenses_present + tenses_future
        is_past_event = True in [" "+w+p in spaced_line for p in punctuations for w in tenses_past]
        is_future_event = True in [" "+w+p in spaced_line for p in punctuations for w in tenses_future]
        is_event = True in [" "+w+p in spaced_line for p in punctuations for w in event_tenses]
        
        if " will be " in line:  # A COMMON FUTURE EXPRESSION NOT COVERED BY ABOVE
            line = line.replace(" will be ", " is ")
            is_future_event = True
            is_event = True
            
        non_eternal_marker = ":/:" if is_future_event else (":\\:" if is_past_event else ":|:")
        
        if self.tense_from_sentence:
            self.eternal = not is_event
            for punc in punctuations:
                for tense_word in event_tenses:
                    if " "+tense_word+punc in spaced_line:
                        line = ((line + " ").replace(" "+tense_word+punc, "")).lstrip().rstrip()
        
        # Postag and bring it into canonical representation using Wordnet lemmatizer
        self.sentence = " " + line.replace("!", "").replace("?", "").replace(".", "").replace(",", "").replace(" not ", " ") + " "
        s_and_T = self.sentence_and_types(self.sentence)
        self.sentence = s_and_T[0]  # canonical sentence (with lemmatized words)
        typetext = s_and_T[1]  # " DET_1 ADJ_1 NOUN_1 ADV_2 VERB_2 DET_2 ADJ_2 NOUN_2 ADP_3 DET_3 ADJ_3 NOUN_3 "
        
        self.word_type = dict(zip(typetext.split(" "), self.sentence.split(" ")))  # mappings like cat -> NOUN_1
        self.type_word = dict(zip(self.sentence.split(" "), typetext.split(" ")))  # mappings like NOUN1 -> cat
        
        # Transformed typetext taking syntactical relations and represent relations into account
        (self.typetext_reduced, _) = self.reduce_typetext(typetext)
        (typetext_narsese, _) = self.reduce_typetext(typetext, apply_statement_represent=True)
        (typetext_concrete, truth) = self.reduce_typetext(typetext, apply_statement_represent=True, apply_term_represent=True, suppress_output=False)
        
        if self.verbose:
            results.append("//Lemmatized sentence: " + self.sentence)
            results.append("//Typetext: " + typetext)
            results.append("//Typetext reduced:" + self.typetext_reduced)
            results.append("//Typetext Narsese:" + typetext_narsese)
        
        # Check if representations need grammar learning
        input_valid = True
        typetext_split = [x.strip() for x in typetext_concrete.split(" , ") if x.strip() != ""]
        for y in typetext_split:
            if self.grammar_learning(y):
                input_valid = False
                break
        
        # Output the Narsese events for NARS to consume
        if input_valid:
            for y in typetext_split:
                truth_string = "" if not self.output_truth else " {" + str(truth[0]) + " " + str(truth[1]) + "}"
                statement = "(! " + y + ")" if is_negated else " " + y + " "
                punctuation = "?" if is_question else ("!" if is_goal else ".")
                narsese = (statement
                          .replace(" {What} ", " ?1 ")
                          .replace("=/>", "==>")
                          .replace(" {Who} ", " ?1 ")
                          .replace(" {It} ", " $1 ")
                          .replace(" what ", " ?1 ")
                          .replace(" who ", " ?1 ")
                          .replace(" it ", " $1 ")
                          .strip() + (punctuation + ("" if self.eternal else " " + non_eternal_marker)) + truth_string)
                results.append(narsese)
            
            if len(typetext_split) > 0 and self.thinkcycles is not None:
                results.append(self.thinkcycles)
                
        if self.motivation is not None and line.strip() != "":
            results.append(self.motivation)
            if self.thinkcycles is not None:
                results.append(self.thinkcycles)
                
        return "\n".join(results)

    def process_text(self, text):
        """Process multiple lines of text and return Narsese outputs"""
        lines = text.strip().split('\n')
        results = []
        
        for line in lines:
            result = self.process_line(line)
            if result:
                results.append(result)
                
        return "\n".join(results)

    def interactive(self):
        """Run an interactive session for processing English to Narsese"""
        run = True
        while run:
            try:
                line = input().rstrip("\n")
                result = self.process_line(line)
                if result:
                    print(result)
                    sys.stdout.flush()
            except EOFError:
                break
            except KeyboardInterrupt:
                break

def main():
    """Main function to run the script directly"""
    verbose = "verbose" in sys.argv
    output_truth = "OutputTruth" in sys.argv
    eternal_output = "EternalOutput" in sys.argv
    
    converter = EnglishToNarsese(
        verbose=verbose,
        output_truth=output_truth,
        eternal_output=eternal_output
    )
    
    converter.interactive()

if __name__ == "__main__":
    main()