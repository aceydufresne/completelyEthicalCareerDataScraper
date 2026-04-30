from pypdf import PdfReader
import kagglehub
import pandas as pd
#replacing nltk with tdf module
import nltk
from nltk.corpus import stopwords
import math
from listingSet import runScraper
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def uploadRes(resPath):
    reader = PdfReader(resPath)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    lines = text.split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def cleanSet(df, tot):
    indSet = tot
    tempD = {}
    custStop = ['\\xe2\\x80\\x93','human', '•', '–']
    stopWords = set(stopwords.words("english"))
    words = df.split()
    for word in words:
        word = word.lower()
        if word in custStop:
            continue
        if word not in stopWords:
            if word in indSet and word in tempD:
                indSet[word] += 1
                tempD[word] += 1
            elif word not in indSet and word not in tempD:
                tempD[word] = 1
                indSet[word] = 1
            elif word in indSet and word not in tempD:
                indSet[word] += 1
                tempD[word] = 1
            elif word not in indSet and word in tempD:
                tempD[word] += 1
                indSet[word] = 1
        else:
            continue
    return indSet,tempD

def findTF(inRes, tot, loc, docSets):
    totalProb = {}
    for term in inRes:
        tempCount = loc[term]
        if tempCount == 0:
            tempCount = 1
        tf = tempCount / len(inRes)
        N = len(docSets)
        i = 0
        for doc in docSets:
            if term in doc:
                i +=1
            else:
                continue
        if i == 0:
            i+=1
        idf = math.log(N/i)
        tfIDF = tf * idf
        totalProb[term] = tfIDF
    return totalProb




def sortListing(listArray):
    
    allVecs = []
    lineTF = []
    lineCount = {}
    
    sumSec = []
    skillSec = []
    eduSec = []
    expSec = []
    finalSec = []
    
    for line in listArray:
        embedding = model.encode(line,convert_to_tensor=True)
        allVecs.append(embedding)
        words = line.split()
        N = len(words)
        terms = {}
        
        for word in words:
            terms[word] = terms.get(word, 0) + 1

        for term in terms:
            tempCount = terms[term]
            tf = tempCount / N
            if tf == 0:
                tf += 1
            lineTF.append(tf)
            lineCount[term] = lineCount.get(term, 0) + 1

    skills = {
        'responsibilities': .5,
        
    }
    experience = {
        'qualifications': .5,
        "experience":5.0,
        "work":.3, 
        "employer":.3, 
        "experience":.2, 
        "intern":.5, 
        "internship":.5, 
        "employee":.5, 
        "manager":.4
        
    }
    education = {
        "education": .50,
        "university":.2,
        "bachelor's":.5,
        "bachelors":.5,
        "associate's":.5,
        "associates":.5,
        "master's":.5,
        "masters":.5,
        "college":.2,
        "certificate":.5,
        "degree":.5,
        "acedemic":.2
    }
    summary2 = [
        "professional summary",
        "career objective",
        "personal profile",
        "about me"
    ]
    
    projects2= [
        "projects",
        "software projects",
        "research projects",
        "academic projects"
    ]
    
    skills2= [
        "technical skills",
        "programming languages",
        "software tools",
        "core competencies"
    ]
    
    experience2= [
        "work experience",
        "employment history",
        "professional experience",
        "job history",
        "internship experience"
    ]
    
    education2= [
        "education",
        "academic background",
        "degree information",
        "university studies",
        "college education"
    ]
    
    #noMatch = []
    expEmb = [model.encode(p, convert_to_tensor=True) for p in experience2]
    eduEmb = [model.encode(p, convert_to_tensor=True) for p in education2]
    skillsEmb = [model.encode(p, convert_to_tensor=True) for p in skills2]
    sumEmb = [model.encode(p, convert_to_tensor=True) for p in summary2]
    
    
    for line in listArray:
        words = line.split()
        experienceWeight = 0
        skillsWeight = 0
        educationWeight = 0
        summaryWeight = 0
        noMatch = []
        for word in words:
            word = word.lower()
            if word in experience:
                value = experience.get(word)
                experienceWeight += value
            elif word in skills:
                value = skills.get(word)
                skillsWeight += value
            elif word in education:
                value = education.get(word)
                educationWeight+=value
                
                
            elif word in experience2:
                value = .3
                experienceWeight+=value
            elif word in education2:
                value = .3
                educationWeight+=value
            elif word in skills2:
                value = .3
                skillsWeight+=value
            elif word in summary2:
                value = .3
                summaryWeight+=value
            else:
                
                noMatch.append(word)
                embedding2 = model.encode(word,convert_to_tensor=True)
                expDist = []
                eduDist = []
                skillsDist = []
                sumDist = []
                for a in expEmb:
                    minDistExp = 0
                    
                    #embedding = model.encode(a,convert_to_tensor=True)
                    score = util.cos_sim(a, embedding2).item()
                    expDist.append(score)
                for b in eduEmb:
                    minDistEdu = 0
                    
                    #embedding = model.encode(b,convert_to_tensor=True)
                    score = util.cos_sim(b, embedding2).item()
                    eduDist.append(score)
                for c in skillsEmb:
                    minDistSkills = 0
                    
                    #embedding = model.encode(c,convert_to_tensor=True)
                    score = util.cos_sim(c, embedding2).item()
                    skillsDist.append(score)
                for d in sumEmb:
                    minDistSum = 0
                    #embedding = model.encode(d,convert_to_tensor=True)
                    score = util.cos_sim(d, embedding2).item()
                    sumDist.append(score)
                    
                tempValue = []
                minExpDist = max(expDist) 
                minEduDist = max(eduDist) 
                minSkillsDist = max(skillsDist) 
                minSumDist = max(sumDist)
                tempValue.append(minExpDist)
                tempValue.append(minEduDist)
                tempValue.append(minSkillsDist)
                tempValue.append(minSumDist)
                maxDist = max(tempValue)
                if minExpDist == maxDist:
                    experienceWeight+=.1
                elif minEduDist == maxDist:
                    educationWeight+=.1
                elif minSkillsDist == maxDist:
                    skillsWeight+=.1
                elif minSumDist == maxDist:
                    summaryWeight+=.1
        finalWeights = []
        finalWeights.append(experienceWeight)
        finalWeights.append(educationWeight)
        finalWeights.append(skillsWeight)
        finalWeights.append(summaryWeight)
        finalPrediction = max(finalWeights)
        if experienceWeight == finalPrediction:
            expSec.append(line)
        elif educationWeight == finalPrediction:
            eduSec.append(line)
        elif skillsWeight == finalPrediction:
            skillSec.append(line)
        elif summaryWeight == finalPrediction:
            sumSec.append(line)
    finalSec.append(expSec)
    finalSec.append(eduSec)
    finalSec.append(skillSec)
    finalSec.append(sumSec)
    
    return finalSec


def compareResList(sortedRes, allListing):
    finResEmb = []

    for component in range(4):
        resText = " ".join(sortedRes[component])
        if resText.strip():
            tempResEmb = model.encode(resText, convert_to_tensor=True)
        else:
            tempResEmb = None
        finResEmb.append(tempResEmb)
    allScores = []
    for listing in allListing:
        indScores = []
        for resComp, listComp in zip(finResEmb, listing):
            listText = " ".join(listComp)
            if resComp is None or not listText.strip():
                tempScore = 0
            else:
                listEmb = model.encode(listText, convert_to_tensor=True)
                tempScore = util.cos_sim(resComp, listEmb).item()
            indScores.append(tempScore)
        avg = sum(indScores) / len(indScores)
        passed = avg >= 0.5
        allScores.append((avg, passed, indScores))

    for job, score in zip(allListing, allScores):
        if score[1]:
            print("Passes:")
            print("Average:", score[0])
            print("Component scores:", score[2])
            print(job)
            print("-" * 80)

    return allScores
    
def simGraph(scores, listings, top_n=20):
    plotData = []

    for listing, score in zip(listings, scores):
        avg = score[0]
        title = listing.get("title", "Unknown Job")
        plotData.append({
            "title": title,
            "score": avg
        })

    plotData = sorted(plotData, key=lambda x: x["score"], reverse=True)
    plotData = plotData[:top_n]

    titles = [item["title"] for item in plotData]
    values = [item["score"] for item in plotData]

    plt.figure(figsize=(12, 8))
    plt.barh(titles, values)
    plt.xlabel("Similarity score")
    plt.title(f"Top {top_n} Resume to Job Similarity Scores")
    plt.gca().invert_yaxis()

    for i, value in enumerate(values):
        plt.text(value + 0.002, i, f"{value:.3f}", va="center")

    plt.tight_layout()
    plt.show()
    
    
        
if __name__  == "__main__":
    path = 'Dufresne_Resume_Spring_2026.pdf'
    #path = kagglehub.dataset_download("maitrip/resumes")
    path2 = 'resume_dataset.csv'
    totTokens = {}
    docSets = []
    #nltk.download('stopwords')
    
    
    df = pd.read_csv(path2)
    for i in range(len(df)):
        tempDF = df.iloc[i]["Resume"]
        res, tempDoc = cleanSet(tempDF, totTokens)
        totTokens = res
        docSets.append(tempDoc)
    print(max(docSets[0].values()))
    #print(docSets[0])
    
    reader = PdfReader("Dufresne_Resume_Spring_2026.pdf")
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    inputTok = {}
    res1, resTok = cleanSet(text, inputTok)
    print(max(resTok.values()))
    #print(resTok)
    
    scores = findTF(res1, totTokens, resTok, docSets)
    print(max(scores, key=scores.get))
    
    titles = ['Software Engineer', 'ML Engineer', 'Engineer', 'HR', 'Designing' , 'Management']
    location = 'Kennesaw, GA'
    
    listings, embeddings = runScraper(titles, location)
    formattedRes = uploadRes(path)
    sortedRes = sortListing(formattedRes)
    allListing = []
    for listing in listings:
        lines = listing["description"].split("\n")
        sortedListing = sortListing(lines)
        allListing.append(sortedListing)
    finScores = compareResList(sortedRes, allListing)
    simGraph(finScores, listings, top_n = 20)
