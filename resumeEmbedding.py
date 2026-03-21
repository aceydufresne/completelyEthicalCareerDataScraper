from pypdf import PdfReader
import re
from sentence_transformers import SentenceTransformer, util
import pyap
from geopy.geocoders import Nominatim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch

def uploadRes(name,resPath):
    reader = PdfReader(resPath)
    print(len(reader.pages))
    page = reader.pages[0]
    text = page.extract_text()
    #print(text)
    #careerTitle(name,text)
    #parseRes(text)
    return text
    
def parseRes(text):
    #iterate line by line
    #make a prediction what this line represents:
    window = []
    lines = text.split('\n')
    education = []
    experience = []
    skills = []
    location = []
    projects = []
    summary = []
    title = []
    address = None
    embeddings = []
    
    educationKeywords = {
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
    educationBias = 0.0
    experienceKeywords = {
        "experience":5.0,
        "work":.3, 
        "employer":.3, 
        "experience":.2, 
        "intern":.5, 
        "internship":.5, 
        "employee":.5, 
        "manager":.4
    }
    experienceBias = 0.0
    
    locationPattern = [
        r'^[A-Z][a-zA-Z\s]+,\s?[A-Z]{2}$',
        r'^[A-Z][a-zA-Z\s]+,\s?[A-Z][a-z]+$',
        r'^\d+\s+[A-Za-z0-9\s]+\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln)\,?\s+[A-Z][a-zA-Z\s]+\,?\s+[A-Z]{2}\s+\d{5}$',
        r'^\d{5}(-\d{4})?$',
    ]
    locationBias = {
        "usa":1.0,
        "United States": .3,
        
    }
    
    states = [
    "alabama", "alaska", "arizona", "arkansas", "california",
    "colorado", "connecticut", "delaware", "florida", "georgia",
    "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland",
    "massachusetts", "michigan", "minnesota", "mississippi", "missouri",
    "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming"
]
    
    abbr = [
    "al","ak","az","ar","ca","co","ct","de","fl","ga",
    "hi","id","il","in","ia","ks","ky","la","me","md",
    "ma","mi","mn","ms","mo","mt","ne","nv","nh","nj",
    "nm","ny","nc","nd","oh","ok","or","pa","ri","sc",
    "sd","tn","tx","ut","vt","va","wa","wv","wi","wy"
]
    
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
    
    #lists of vectors to compare
    allVec = []
    projectsVec = []
    summaryVec = []
    skillsVec = []
    experienceVec = []
    educationVec = []
    
    for i in projects2:
        tempEm = model.encode(i,convert_to_tensor=True)
        projectsVec.append(('projects',tempEm))
    for x in summary2:
        tempEm = model.encode(x,convert_to_tensor=True)
        summaryVec.append(('summary',tempEm))
    for y in skills2:
        tempEm = model.encode(y,convert_to_tensor=True)
        skillsVec.append(('skills',tempEm))
    for j in experience2:
        tempEm = model.encode(j,convert_to_tensor=True)
        experienceVec.append(('experience',tempEm))
    for a in education2:
        tempEm = model.encode(a,convert_to_tensor=True)
        educationVec.append(('education',tempEm))
    allVec.append(projectsVec)
    allVec.append(summaryVec)
    allVec.append(skillsVec)
    allVec.append(experienceVec)
    allVec.append(educationVec)
    
    wrapWeightEdu = 0.0
    wrapWeightExp = 0.0
    for line in lines:
        #scoring that decides where to add a line into
        #and a bias for keywords that wont show up in
        #sentence_transformers
        addresses = pyap.parse(line, country='US')
        line = line.strip()
        
        for pat in locationPattern:
            if re.match(pat, line):
                location.append(line)
                
        embedding = model.encode(line,convert_to_tensor=True)
        embeddings.append((line,embedding))
        
        words = line.lower().split()
        for word in words:
            if word in locationBias:
                location.append(line)
            if word in states:
                location.append(line)
            #if word in abbr:
                #location.append(line)
            bias = 0
            if word in educationKeywords:
                bias = educationKeywords.get(word)
                #print(bias)
                educationBias+=bias
            if word in experienceKeywords:
                bias = experienceKeywords.get(word)
                #print(bias)
                experienceBias+=bias
            if word in education2:
                if line in education:
                    break
                elif line not in education:
                    education.append(line)
        
            if word in experience2:
                if line in experience:
                    break
                elif line not in experience:
                    experience.append(line)
            if word in skills2:
                if line in skills:
                    break
                elif word not in skills:
                    skills.append(line)
            if word in projects:
                if line in projects:
                    break
                elif line not in projects:
                    projects.append(line)
                    
            if word in summary2:
                if line in summary:
                    break
                elif line not in summary:
                    summary.append(line)
        if experienceBias >= .2:
            experience.append(line)
            #need a wrap around weight to increase the likehood
            #the next line will also be apart of the set
        elif educationBias >= .2:
            education.append(line)
            
        experienceBias = 0.0
        educationBias = 0.0
    #print(experience)
    #print(education)
    #print(location)
    #print(addresses)
    #print(projects)
    #print(summary)
    #print(skills)
    #print(embeddings)
    #print(len(embeddings[0]))
    
    geolocator = Nominatim(user_agent="AddressChecker")
    for locations in location:
        #if this is a real address save as location
        if geolocator.geocode(locations):
            address=locations
            print(address)
    
    sections = findCos(embeddings,allVec)
    return sections

def findCos(embeddings, allVec):
    #list of lists
    projectSec = []
    summarySec = []
    skillSec = []
    experienceSec = []
    educationSec = []
    unsorted = []
    
    sectionLists = [
        ("projects", projectSec),
        ("summary", summarySec),
        ("skills", skillSec),
        ("experience", experienceSec),
        ("education", educationSec)
    ]
    
    
    for line, embVec in embeddings:
        
        bestSelScore = -1
        bestSel = None
        for ind, compVec in enumerate(allVec):
            currentSecName, currentList = sectionLists[ind]
            #just in case none of the sections match well
            tempScore = -1
            for descr, indVec in compVec:
                score = util.cos_sim(embVec, indVec).item()
                
                if score > tempScore:
                    tempScore = score
            if tempScore > bestSelScore:
                bestSelScore = tempScore
                bestSel = ind
        if bestSelScore>.35:
            sectionLists[bestSel][1].append((line,bestSelScore))
        else:
            unsorted.append(("Unsorted", line))
        
    print("Proj")
    print(projectSec)
    print("sum")
    print(summarySec)
    print("skill")
    print(skillSec)
    print("exp")
    print(experienceSec)
    print("edu")
    print(educationSec)
    print("the rest")
    print(unsorted)
    return {
    "projects": projectSec,
    "summary": summarySec,
    "skills": skillSec,
    "experience": experienceSec,
    "education": educationSec,
    "unsorted": unsorted
}
                
def careerTitle(name, text):
    careerTitles = [
    "Software Engineer",
    "Data Analyst",
    "Data Scientist",
    "Machine Learning Engineer",
    "Computer Vision Engineer",
    "Web Developer",
    "Backend Developer",
    "Frontend Developer",
    "Full Stack Developer",
    "Business Analyst",
    "Financial Analyst",
    "Project Manager",
    "Cybersecurity Analyst",
    "Systems Administrator"
]
    examplePos = []
    checkPos = []
    
    for career in careerTitles:
        posEmb = model.encode(career,convert_to_tensor=True)
        examplePos.append((career, posEmb))
    
    name = name.lower()
    window = []
    foundName = False
    lines = text.split("\n")
    for i in range(len(lines)):
        line = lines[i].lower()
        if name in line:
            #position is more likely to be near
            #applicants name
            foundName = True
            lr = max(0, i-3)
            ur = min(len(lines), i+3)
            placeholder = i
            for x in range(lr,ur):
                temp = lines[x].strip()
                if temp:
                    emb = model.encode(temp,convert_to_tensor=True)
                    checkPos.append((temp, emb))
            break
    
    if foundName == False:
        print("Error invalid Name")
        return None
    
    bestTitle = None
    bestScore = -1
    #check similiarity between the vectors
    for tit,vec in checkPos:
        for checkTit, checkEmb in examplePos:
            score1 = util.cos_sim(vec, checkEmb).item()    
            if score1 > bestScore:
                bestScore = score1
                bestTitle = checkTit
                
    print(bestScore)
    print(bestTitle)
    return bestScore, bestTitle










##not going to stay, just to test how the agent would do
def example():
    about = """Candescent is the leading cloud-based digital banking solutions provider for financial institutions. 
We are transforming digital banking with intelligent, cloud-powered solutions that connect account opening, digital banking, 
and branch experiences for financial institutions.

Success here requires flexibility in a fast-paced environment, a client-first mindset, and a commitment to delivering consistent,
reliable results as part of a performance-driven, values-led team.

Candescent is currently searching for a Software Engineer who is passionate about creating world-class software experiences.
"""

    skills = """Implement Features and Deliver Production-Ready Code

Develop and deploy secure, reliable features
Create technical documentation, system diagrams, and debugging reports
Optimize performance and monitor key metrics

Product Strategy, Vision, and Planning

Participate in agile planning events with technical insights and feasibility analysis
Collaborate with Product Owners, Designers, and Engineers to translate business needs into technical solutions
Provide input on technology enhancements, emerging tools, and product gaps to shape the roadmap
Support definition of acceptance criteria for development and testing

Product Design and Development

Design, build, and maintain software solutions in an agile environment
Write clean, scalable code aligned with best practices and standards
Collaborate with peers to implement user stories, resolve issues, and improve functionality
Engage in agile ceremonies such as daily scrums and demos
Create proof-of-concepts and run experiments to guide technical decisions
Conduct peer code reviews for quality and knowledge sharing
Continuously improve team processes and workflows
Stay current with emerging technologies, frameworks, and industry trends

Performance Measurement and Optimization

Track delivery metrics such as lead time and deployment frequency
Identify and implement improvements to enhance efficiency
"""

    education = """Bachelor’s degree in computer science, Information Technology, or equivalent
4+ years of experience in software development using Java
Strong foundation in data structures, algorithms, and concurrent programming
Expertise in designing and troubleshooting transactional systems
Experience with microservices architecture, Java EE, Spring Boot, Spring Cloud, Hibernate, Oracle, PostgreSQL, BigTable, BigQuery, NoSQL, Git, IntelliJ IDEA, Pub/Sub, Data Flow
Familiarity with native and hybrid cloud environments and Agile development
Proficient in Python or Java, multi-tenant cloud technologies, and tools like Jira
Skilled in translating user stories into scalable, user-centric solutions
Strong collaboration and communication skills for cross-functional teamwork
Analytical problem-solver with attention to detail and structured thinking
Experience with Java IDEs (Eclipse, IntelliJ), application servers (Tomcat), scripting languages (JSON, XML, YAML, Terraform), Git, Maven"""

    location = "Atlanta, GA"

    about_lines = [x.strip() for x in about.split("\n") if x.strip()]
    skills_lines = [x.strip() for x in skills.split("\n") if x.strip()]
    education_lines = [x.strip() for x in education.split("\n") if x.strip()]
    location_lines = [location.strip()]

    about_emb = model.encode(about_lines, convert_to_tensor=True)
    skills_emb = model.encode(skills_lines, convert_to_tensor=True)
    education_emb = model.encode(education_lines, convert_to_tensor=True)
    location_emb = model.encode(location_lines, convert_to_tensor=True)

    about_emb = F.normalize(about_emb, p=2, dim=1)
    skills_emb = F.normalize(skills_emb, p=2, dim=1)
    education_emb = F.normalize(education_emb, p=2, dim=1)
    location_emb = F.normalize(location_emb, p=2, dim=1)

    return {
        "about": (about_lines, about_emb),
        "skills": (skills_lines, skills_emb),
        "education": (education_lines, education_emb),
        "location": (location_lines, location_emb)
    }
def compare_sections(resume_data, job_data):
    results = {}

    for section in ["skills", "education", "location"]:
        res_lines, res_emb = resume_data.get(section, ([], None))
        job_lines, job_emb = job_data.get(section, ([], None))

        if res_emb is None or job_emb is None or len(res_lines) == 0:
            continue

        scores = util.cos_sim(res_emb, job_emb)

        section_results = []

        for i, res_line in enumerate(res_lines):
            best_idx = scores[i].argmax().item()
            best_score = scores[i][best_idx].item()

            section_results.append({
                "resume_line": res_line,
                "job_match": job_lines[best_idx],
                "score": best_score
            })

        results[section] = section_results

    return results   

def prepare_resume_for_compare(sections):
    resume_data = {}

    for key in ["skills", "education", "location"]:
        sec = sections.get(key, [])

        if not sec:
            continue

        # extract just the text
        lines = [item[0] for item in sec]

        # embed + normalize
        emb = model.encode(lines, convert_to_tensor=True)
        emb = F.normalize(emb, p=2, dim=1)

        resume_data[key] = (lines, emb)

    return resume_data
def plot_embeddings_scatter(resume_data, job_data):
    all_lines = []
    all_embs = []
    colors = []

    # Color map
    color_map = {
        "resume": "blue",
        "job": "red"
    }

    # ---- Resume ----
    for section_name, (lines, emb) in resume_data.items():
        if emb is None or len(lines) == 0:
            continue

        emb = F.normalize(emb, p=2, dim=1)

        for i, line in enumerate(lines):
            all_lines.append(line)
            all_embs.append(emb[i].cpu())
            colors.append(color_map["resume"])

    # ---- Job ----
    for section_name, (lines, emb) in job_data.items():
        if emb is None or len(lines) == 0:
            continue

        emb = F.normalize(emb, p=2, dim=1)

        for i, line in enumerate(lines):
            all_lines.append(line)
            all_embs.append(emb[i].cpu())
            colors.append(color_map["job"])

    if len(all_embs) < 2:
        print("Not enough points to plot.")
        return

    # Stack embeddings
    X = torch.stack(all_embs).numpy()

    # PCA → 2D
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(X)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c=colors, alpha=0.7)

    # Annotate (lighter)
    for i, txt in enumerate(all_lines):
        short_txt = txt[:25].replace("\n", " ")
        plt.annotate(short_txt, 
                     (points_2d[i, 0], points_2d[i, 1]), 
                     fontsize=7, alpha=0.6)

    plt.title("Resume vs Job Embeddings (PCA Projection)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    # Legend
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(label='Resume')
    red_patch = mpatches.Patch(label='Job')
    plt.legend(handles=[blue_patch, red_patch])

    plt.show()
def plot_similarity_heatmap(resume_data, job_data, section="skills"):
    if section not in resume_data or section not in job_data:
        print(f"Section '{section}' not found in both resume_data and job_data.")
        return

    res_lines, res_emb = resume_data[section]
    job_lines, job_emb = job_data[section]

    if res_emb is None or job_emb is None or len(res_lines) == 0 or len(job_lines) == 0:
        print(f"Not enough data to plot heatmap for section '{section}'.")
        return

    # normalize just to be safe
    res_emb = F.normalize(res_emb, p=2, dim=1)
    job_emb = F.normalize(job_emb, p=2, dim=1)

    sim_matrix = util.cos_sim(res_emb, job_emb).cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.imshow(sim_matrix, aspect="auto")
    plt.colorbar(label="Cosine Similarity")

    plt.xticks(
        ticks=np.arange(len(job_lines)),
        labels=[x[:25].replace("\n", " ") for x in job_lines],
        rotation=45,
        ha="right"
    )
    plt.yticks(
        ticks=np.arange(len(res_lines)),
        labels=[x[:25].replace("\n", " ") for x in res_lines]
    )

    plt.title(f"Similarity Heatmap: Resume vs Job ({section})")
    plt.xlabel("Job Lines")
    plt.ylabel("Resume Lines")
    plt.tight_layout()
    plt.show()
def plot_best_similarity_scores(resume_data, job_data, section="skills"):
    if section not in resume_data or section not in job_data:
        print(f"Section '{section}' not found in both resume_data and job_data.")
        return

    res_lines, res_emb = resume_data[section]
    job_lines, job_emb = job_data[section]

    if res_emb is None or job_emb is None or len(res_lines) == 0 or len(job_lines) == 0:
        print(f"Not enough data to plot similarity scores for section '{section}'.")
        return

    # normalize
    res_emb = F.normalize(res_emb, p=2, dim=1)
    job_emb = F.normalize(job_emb, p=2, dim=1)

    sim_matrix = util.cos_sim(res_emb, job_emb)

    best_scores = []
    best_matches = []

    for i in range(sim_matrix.shape[0]):
        best_idx = sim_matrix[i].argmax().item()
        best_score = sim_matrix[i][best_idx].item()

        best_scores.append(best_score)
        best_matches.append(job_lines[best_idx])

    x_labels = [line[:25].replace("\n", " ") for line in res_lines]

    plt.figure(figsize=(12, 8))
    plt.scatter(range(len(best_scores)), best_scores)

    for i, score in enumerate(best_scores):
        plt.annotate(f"{score:.2f}", (i, best_scores[i]), fontsize=8)

    plt.xticks(
        ticks=range(len(x_labels)),
        labels=x_labels,
        rotation=45,
        ha="right"
    )

    plt.ylim(0, 1)
    plt.title(f"Best Similarity Scores per Resume Line ({section})")
    plt.xlabel("Resume Lines")
    plt.ylabel("Best Cosine Similarity")
    plt.tight_layout()
    plt.show()

    return list(zip(res_lines, best_matches, best_scores))
def plot_similarity_scatter_with_links(resume_data, job_data, section="skills"):
    if section not in resume_data or section not in job_data:
        print(f"Section '{section}' not found in both resume_data and job_data.")
        return

    res_lines, res_emb = resume_data[section]
    job_lines, job_emb = job_data[section]

    if res_emb is None or job_emb is None or len(res_lines) == 0 or len(job_lines) == 0:
        print(f"Not enough data to plot similarity scatter for section '{section}'.")
        return

    res_emb = F.normalize(res_emb, p=2, dim=1)
    job_emb = F.normalize(job_emb, p=2, dim=1)

    all_embs = torch.cat([res_emb, job_emb], dim=0).cpu().numpy()
    all_labels = [f"R: {x[:20]}" for x in res_lines] + [f"J: {x[:20]}" for x in job_lines]

    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(all_embs)

    res_points = points_2d[:len(res_lines)]
    job_points = points_2d[len(res_lines):]

    sim_matrix = util.cos_sim(res_emb, job_emb)

    plt.figure(figsize=(12, 8))

    # resume points
    plt.scatter(res_points[:, 0], res_points[:, 1], label="Resume", alpha=0.7)

    # job points
    plt.scatter(job_points[:, 0], job_points[:, 1], label="Job", alpha=0.7)

    # annotate
    for i, txt in enumerate(res_lines):
        plt.annotate(txt[:20], (res_points[i, 0], res_points[i, 1]), fontsize=7)
    for i, txt in enumerate(job_lines):
        plt.annotate(txt[:20], (job_points[i, 0], job_points[i, 1]), fontsize=7)

    # draw links to best match
    for i in range(sim_matrix.shape[0]):
        best_idx = sim_matrix[i].argmax().item()
        x1, y1 = res_points[i]
        x2, y2 = job_points[best_idx]
        plt.plot([x1, x2], [y1, y2], linestyle="--", alpha=0.5)

    plt.title(f"Similarity Scatter Plot with Best Matches ({section})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.show()




    
    
if __name__  == "__main__":
    #nltk.download("stopwords")
    path = 'https://www.linkedin.com/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2F'
    path2 = 'https://www.worksourceatlanta.org/job-trends/high-demand-occupations/'
    resPath = 'Dufresne_Resume_Spring_2026.pdf'
    dsRes = 'resDatasets/'
    
    name = input("Enter applicants name:")
    
    
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    text = uploadRes(name,resPath)
    position = careerTitle(name,text)
    #address = parseRes(text)
    sections = parseRes(text)
    
    
    resume_data= prepare_resume_for_compare(sections)
    job_data = example()
    results = compare_sections(resume_data, job_data)
    plot_embeddings_scatter(resume_data, job_data)
    plot_similarity_heatmap(resume_data, job_data, section="skills")
    plot_best_similarity_scores(resume_data, job_data, section="skills")
    plot_similarity_scatter_with_links(resume_data, job_data, section="skills")