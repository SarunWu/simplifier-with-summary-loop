import os
# import textstat
import textstatjk as jk
import spacy
from nltk.corpus import wordnet as wn

dc = jk.textstatistics()
sp = spacy.load("en_core_web_sm")


# find NER in the doc

def findNER(doc):
    doc = sp(doc)
    nerlist = doc.ents
    # print(nerlist)
    # spacy.displacy.render(doc, style="ent",jupyter=True)
    return nerlist


# print NER in the doc

def showNER(doc):
    doc = sp(doc)
    nerlist = doc.ents
    # nerlist = findNER(doc1)
    print(nerlist)
    spacy.displacy.render(doc, style="ent", jupyter=True)


def findsynset(diffword, wordpos):
    allword = set()
    easyword1 = []
    easyword2 = []
    easyword3 = []

    # look for the synset according to POS
    if wordpos == "NOUN":
        allsynset = wn.synsets(diffword, pos=wn.NOUN)
    elif wordpos == "VERB":
        allsynset = wn.synsets(diffword, pos=wn.VERB)
    elif wordpos == "ADJ":
        allsynset = wn.synsets(diffword, pos=wn.ADJ)
    elif wordpos == "ADV":
        allsynset = wn.synsets(diffword, pos=wn.ADV)
    else:
        return diffword

    # extract all word in synset to a set of allword
    if len(allsynset) > 0:
        for i in range(len(allsynset)):
            for j in range(len(allsynset[i].lemmas())):
                # print(i,j,allsynset[i].lemmas()[j].name())
                allword.add(allsynset[i].lemmas()[j].name())

    # filter out the diff word in the set to 3 tier
    for i in allword:
        if "-" in i:
            easyword2.append(i)
        if "_" in i:
            easyword3.append(i)
        if not dc.dcis_difficult_word(i):
            easyword1.append(i)
    
    print("Allword set is: ", allword)

    # select word to return
    if len(easyword1) > 0:
        return easyword1[0]
    elif len(easyword2) > 0:
        return easyword2[0]
    elif len(easyword3) > 0:
        return easyword3[0]
    else:
        return diffword


"""  
    print("candidate ", allword, "\n")
    print("synset ", allsynset, "\n")
    print("diffword ", diffword, "\n")        
    print("eaey tier 1 ", easyword1, "\n")
    print("eaey tier 2 ", easyword2, "\n")
    print("eaey tier 3 ", easyword3, "\n")
"""


def simplify(doc):
    doc = sp(doc)
    simdoc = []
    # print(len(doc))
    for i in range(len(doc)):
        word = doc[i].text
        wordlemma = doc[i].lemma_.lower()
        wordpos = doc[i].pos_
        wordner = doc[i].ent_iob_
        # print(word, wordlemma, wordpos, wordner)

        # check if token is NER
        if doc[i].ent_iob_ != "O":
            simdoc.append(word)
            # print(i, simdoc)
            continue

        # check if word is in easy list
        if not dc.dcis_difficult_word(wordlemma):
            simdoc.append(word)
            # print(i, simdoc)
            continue

        # simplify if diff word
        if dc.dcis_difficult_word(wordlemma):
            simword = findsynset(word, wordpos)
            simdoc.append(simword)
            # print(i, simdoc)

    return simdoc

    # print(i, simdoc)
    # print(doc[i].text, doc[i].pos_)


def diffword(doc):
    doc = sp(doc)
    diffdoc = []
    # print(len(doc))
    for i in range(len(doc)):
        word = doc[i].text
        wordlemma = doc[i].lemma_.lower()
        wordpos = doc[i].pos_
        wordner = doc[i].ent_iob_
        # print(word, wordlemma, wordpos, wordner)

        # check if token is NER
        if doc[i].ent_iob_ != "O":
            continue

        # diff word
        if dc.dcis_difficult_word(wordlemma):
            diffdoc.append(word)
            # print(i, diffdoc)

    print("Difficult words\n")
    print(diffdoc)

    return diffdoc


def printsim(doc):
    finaltext = ""
    for i in doc:
        if "_" in i:
            # print(i)
            i = i.replace("_", " ")
            # print(i)
        finaltext += i + " "
    # print(finaltext)
    return finaltext


# --------------------------------------------------------------------------------------------------------------------------------------------

## PEGASUS MODEL

# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# import torch

# model_name = "google/pegasus-cnn_dailymail"
# # model_name = "google/pegasus-xsum"

# device = "cuda" if torch.cuda.is_available() else "cpu"

# print("Downloading model")
# model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
# tokenizer = PegasusTokenizer.from_pretrained(model_name)

# ARTICLE_TO_SUMMARIZE = ("""
# -LRB- CNN -RRB- The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday , a step that gives the court jurisdiction over alleged crimes in Palestinian territories . The formal accession was marked with a ceremony at The Hague , in the Netherlands , where the court is based . The Palestinians signed the ICC 's founding Rome Statute in January , when they also accepted its jurisdiction over alleged crimes committed in the occupied Palestinian territory , including East Jerusalem , since June 13 , 2014 . '' Later that month , the ICC opened a preliminary examination into the situation in Palestinian territories , paving the way for possible war crimes investigations against Israelis . As members of the court , Palestinians may be subject to counter-charges as well . Israel and the United States , neither of which is an ICC member , opposed the Palestinians ' efforts to join the body . But Palestinian Foreign Minister Riad al-Malki , speaking at Wednesday 's ceremony , said it was a move toward greater justice . As Palestine formally becomes a State Party to the Rome Statute today , the world is also a step closer to ending a long era of impunity and injustice , '' he said , according to an ICC news release . Indeed , today brings us closer to our shared goals of justice and peace . Judge Kuniko Ozaki , a vice president of the ICC , said acceding to the treaty was just the first step for the Palestinians . As the Rome Statute today enters into force for the State of Palestine , Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute . These are substantive commitments , which can not be taken lightly , '' she said . Rights group Human Rights Watch welcomed the development . Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure , and countries that support universal acceptance of the court 's treaty should speak out to welcome its membership , '' said Balkees Jarrah , international justice counsel for the group . What 's objectionable is the attempts to undermine international justice , not Palestine 's decision to join a treaty to which over 100 countries around the world are members . In January , when the preliminary ICC examination was opened , Israeli Prime Minister Benjamin Netanyahu described it as an outrage , saying the court was overstepping its boundaries . The United States also said it strongly disagreed with the court 's decision . As we have said repeatedly , we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC , the State Department said in a statement . It urged the warring sides to resolve their differences through direct negotiations. We will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace , '' it said . But the ICC begs to differ with the definition of a state for its purposes and refers to the territories as Palestine . While a preliminary examination is not a formal investigation , it allows the court to review evidence and determine whether to investigate suspects on both sides . Prosecutor Fatou Bensouda said her office would conduct its analysis in full independence and impartiality . The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead . The inquiry will include alleged war crimes committed since June . The International Criminal Court was set up in 2002 to prosecute genocide , crimes against humanity and war crimes . CNN 's Vasco Cotovio , Kareem Khadder and Faith Karimi contributed to this report .
# """)
# inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="pt", truncation=True).to(device)

# # Generate Summary
# summary_ids = model.generate(inputs["input_ids"], num_beams=2, max_length=50)
# post_sum = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

# simplified_text = simplify(post_sum)
# print(post_sum)
# print(simplified_text)

# simplified_summary = printsim(simplified_text)

#------------------

# print(simplified_summary)

# to RUN: python Generate_summary_NLP.py
