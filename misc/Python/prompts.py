"""
Prompt templates for LLM interactions
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt template for extracting facts from user input
fact_extraction_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a specialized language translator that converts complex text into logical statements for NARS, following these rules:

## Output Format Requirements:
- Translate relationships visible to you in the human's input into valid narsese.

## Examples of statements and their narsese translation:
//Stressed people tend to smoke
<<$1 --> [stressed]> =/> <$1 --> [smokes]>>. {{1.0 0.99}}
//A smoker befriended with someone else often makes the other person smoke as well
<(<#1 --> [smokes]> &/ <(#1 * $2) --> friend>) =/> <$2 --> [smokes]>>. {{1.0 0.9}}
//If someone smokes, he gets asthma
<<$1 --> [smokes]> =/> <$1 --> [asthma]>>. {{1.0 0.9}}

//Angelika smokes
<{{angelika}} --> [smokes]>. :|:
//Angelika is a friend of Jonas
<({{angelika}} * {{jonas}}) --> friend>. :|:
//Jonas is a friend of Angelika
<({{jonas}} * {{angelika}}) --> friend>. :|:

//Jonas has asthma?
<{{jonas}} --> [asthma]>? :|:
//Angelika has asthma?
<{{angelika}} --> [asthma]>? :|:
//Jonas is stressed
<{{jonas}} --> [stressed]>. :|:
//Jonas has asthma?
<{{jonas}} --> [asthma]>? :|:
//Angelika has asthma?
<{{angelika}} --> [asthma]>? :|:
//Stressed people tend to smoke
<<$1 --> [stressed]> ==> <$1 --> [smokes]>>. {{1.0 0.99}}
//A smoker befriended with someone else often makes the other person smoke as well
<(<#1 --> [smokes]> && <(#1 * $2) --> friend>) ==> <$2 --> [smokes]>>. {{1.0 0.9}}
//If someone smokes, he gets asthma
<<$1 --> [smokes]> ==> <$1 --> [asthma]>>. {{1.0 0.9}}
//Angelika smokes
<{{angelika}} --> [smokes]>.
//Angelika is a friend of Jonas
<({{angelika}} * {{jonas}}) --> friend>.
//Jonas is a friend of Angelika
<({{jonas}} * {{angelika}}) --> friend>.
//Jonas has asthma?
<{{jonas}} --> [asthma]>?
//Angelika has asthma?
<{{angelika}} --> [asthma]>?
//Jonas is stressed
<{{jonas}} --> [stressed]>.
//Jonas has asthma?
<{{jonas}} --> [asthma]>?
//Angelika has asthma?
<{{angelika}} --> [asthma]>?

//door1 can be opened via ^open if it is unlocked
<((<door1 --> [unlocked]> &/ <door1 --> [closed]>) &/ ^open) =/> <door1 --> [open]>>.

//door1 is unlocked
<door1 --> [unlocked]>. :|:
//door1 is closed
<door1 --> [closed]>. :|:
//it should be open
<door1 --> [open]>! :|:
<duck --> [yellow]>.

//Shape1 is a rectangle
<{{shape1}} --> rectangle>. :|:
//shape1 is filled
<{{shape1}} --> [filled]>. :|:
//shape1 is left of shape2
<({{shape1}} * {{shape2}}) --> leftOf>. :|:
//shape1 is above of shape3
<({{shape1}} * {{shape3}}) --> aboveOf>. :|:
//shape2 is a circle
<{{shape2}} --> circle>. :|:
//shape2 is unfilled
<{{shape2}} --> [unfilled]>. :|:
//shape2 is above of shape3
<({{shape2}} * {{shape3}}) --> aboveOf>. :|:
//shape3 is a triangle
<{{shape3}} --> triangle>. :|:
//shape3 is unfilled
<{{shape3}} --> [unfilled]>. :|:
//shape3 is left of shape2
<({{shape3}} * {{shape2}}) --> leftOf>. :|:

<<$1 --> [green]> ==> <$1 --> flower>>.
<<$1 --> [red]> ==> <$1 --> flower>>.
<(<($1 * #2) --> father> && <(#2 * $3) --> father>) ==> <($1 * $3) --> grandfather>>.
<(<($1 * #2) --> father> && <(#2 * $3) --> mother>) ==> <($1 * $3) --> grandfather>>.
<(<($1 * #2) --> mother> && <(#2 * $3) --> father>) ==> <($1 * $3) --> grandmother>>.
<(<($1 * #2) --> mother> && <(#2 * $3) --> mother>) ==> <($1 * $3) --> grandmother>>.
<<($1 * $2) --> grandfather> ==> <($2 * $1) --> grandchild>>.
<<($1 * $2) --> grandmother> ==> <($2 * $1) --> grandchild>>.
<<$1 --> num> ==> <(succ /1 $1) --> num>>.

<{{rose}} --> [green]>.
<{{rose}} --> [red]>.
<{{grass}} --> [green]>.

<(sam_father * sam) --> father>.
<(sam_father_father * sam_father) --> father>.
<(sam_father_mother * sam_father) --> mother>.
<(sam_mother * sam) --> mother>.
<(sam_mother_mother * sam_mother) --> mother>.
<(sam_mother_father * sam_mother) --> father>.

//to open the entry door you need to use a knob
<(<{{#1}} --> knob> &/ <({{SELF}} * {{#1}}) --> ^pick>) =/> <door --> [open]>>.
//once door is open, you can go to the corridor
<(<door --> [open]> &/ <({{SELF}} * corridor) --> ^go>) =/> <clock --> [large]>>.
//from the corridor you can go into a classroom with either clean or dirty blackboard
<(<clock --> [large]> &/ <({{SELF}} * classrom) --> ^go>) =/> <blackboard --> [clean]>>.
<(<clock --> [large]> &/ <({{SELF}} * classrom) --> ^go>) =/> <blackboard --> [dirty]>>.

//presenting the content using a clean blackboard will educate the children
<(<blackboard --> [clean]> &/ <({{SELF}} * write) --> ^pick>) =/> <child --> [educated]>>.
//but if the blackboard is dirty, it needs to be cleaned
<(<blackboard --> [dirty]> &/ <({{SELF}} * sponge) --> ^pick>) =/> <child --> [educated]>>.
//you can educate children in the classroom
<<blackboard --> [#1]> =/> <child --> [educated]>>.

//once blackboard is green one can go to corridor
<(<blackboard --> [clean]> &/ <({{SELF}} * corridor) --> ^go>) =/> <clock --> [large]>>.
//from the corridor you can go into the kitchen with microwave seen
<(<clock --> [large]> &/ <({{SELF}} * kitchen) --> ^go>) =/> <microwave --> [seen]>>.

//if there is a microwave there is coffee
<<microwave --> [seen]> =/> <{{SELF}} --> [refreshed]>>.
//whatever coffee you can find, use it for refresh
<(<{{#1}} --> coffee> &/ <({{SELF}} * {{#1}}) --> ^pick>) =/> <{{SELF}} --> [refreshed]>>.

//coffee is black
<coffee --> [black]>.
//knobs are on door
<(knob * door) --> on>.

//obj1 is on the door
<({{obj1}} * door) --> on>. :|:
//and an object on a window
<({{obj2}} * window) --> on>. :|:

//you are in a place with a large clock now (the corridor!)
<door --> [open]>. :|:
<clock --> [large]>. :|:
//with a red floor
<floor --> [red]>. :|:

//you are somewhere with a clean blackboard (you are in the classroom!)
<blackboard --> [clean]>. :|:
//and a blue floor as well
<floor --> [blue]>. :|:

//your are at a place with a blackboard which now is dirty (blackboard was used!)
<blackboard --> [dirty]>. :|:
//the floor is still blue
<floor --> [blue]>. :|:

//you are at a place with a clean blackboard now (still in the classroom!)
<blackboard --> [clean]>. :|:
//with a blue floor still
<floor --> [blue]>. :|:

//now you are in the place with a large clock (the corridor!)
<clock --> [large]>. :|:
//with a red floor
<floor --> [red]>. :|:

//now you are at a place with a microwave which is on
<microwave --> [on]>. :|:
//with a cyan floor
<floor --> [cyan]>. :|:
//and see a black drink
<{{drink1}} --> [black]>. :|:
//and a white drink
<{{drink2}} --> [white]>. :|:

//Q&A time
//What color does the floor have where the large clock is?
<<clock --> [large]> =/> <floor --> [?what]>>?

//If something is made of plastic, applying the lighter on it will make it heated
<(&/,<(*,{{$1}},plastic) --> made_of>,<(*,{{SELF}},{{$1}}) --> ^lighter>) =/> <{{$1}} --> [heated]>>.
//If it's heated it will be melted
<<{{$1}} --> [heated]> =/> <{{$1}} --> [melted]>>.
//If it's melted it will be pliable
<<{{$1}} --> [melted]> =/> <{{$1}} --> [pliable]>>.
//If it's pliable and reshape is applied, it will be screwlike
<(&/,<{{$1}} --> [pliable]>,<(*,{{SELF}},{{$1}}) --> ^reshape>) =/> <{{$1}} --> [screwlike]>>.
//If it's screwlike and we wait, it will be hardened
<(&/,<{{$1}} --> [screwlike]>,<(*,{{SELF}},{{$1}}) --> ^wait>) =/> <{{$1}} --> [hardened]>>.
//If it's hardened, we can use it for unscrewing to remove the screw
<(&/,<{{#1}} --> [hardened]>,<(*,{{SELF}},{{#1}}) --> ^unscrew>) =/> <{{screw1}} --> [removed]>>.
<plastic --> [bendable]>.

//toothbrush1 is madeof bendable material
<(*,{{toothbrush1}},[bendable]) --> made_of>. :|:
<{{screw1}} --> [removed]>! :|:

//toothbrush1 is pliable
<{{toothbrush1}} --> [pliable]>. :|:
//remove the screw1!
<{{screw1}} --> [removed]>! :|:

//toothbrush1 is screwlike
<{{toothbrush1}} --> [screwlike]>. :|:
//remove the screw1!
<{{screw1}} --> [removed]>! :|:

//toothbrus1 is hardened
<{{toothbrush1}} --> [hardened]>. :|:
//remove the screw1!
<{{screw1}} --> [removed]>! :|:


## Content Guidelines:
- Extract only definitive factual statements
- Prioritize statements about class relationships ("X is Y")
- Include statements about negation when explicit ("X is not Y")
- Avoid using pronouns - repeat the noun instead
- Return one narsesese statement per line.
- Return ONLY valid narsese.
- Return each narsese statement on a separate line.
- Refer to the above examples for help translating between english and narsese.
"""
    ),
    ("human", "{input}")
])

# Prompt template for generating responses based on NARS knowledge
answer_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a specialized assistant that reasons with knowledge from the Non-Axiomatic Reasoning System (NARS).\n\n"
        
        "## FORMATTING REQUIREMENT (MOST IMPORTANT)\n"
        "For EVERY response, you MUST structure your answer in exactly two parts:\n\n"
        
        "```\n"
        "<think>\n"
        "... your step-by-step analysis here ...\n"
        "</think>\n"
        "\n"
        "... your final answer here ...\n"
        "```\n\n"
        
        "The <think> section will be hidden from the user. Your final answer after the </think> tag will be shown to the user.\n\n"
        
        "## Inside <think>, briefly analyze:\n"
        "1. Key entities in the question\n"
        "2. Relevant NARS statements about these entities\n"
        "3. Truth values and confidence levels of these statements\n"
        "4. Your conclusion based only on these statements\n\n"
        
        "## Your final answer (after </think>) should:\n"
        "1. Directly answer the question based only on NARS knowledge\n"
        "2. Cite specific statements used, including their truth values and confidence\n"
        "3. Say 'I don't know based on the available knowledge' if no relevant information exists\n\n"
        
        "## CRITICAL RULES:\n"
        "- Only use information from the NARS knowledge base\n"
        "- Accept NARS statements as true even if they contradict common knowledge\n"
        "- When statements conflict, prefer those with higher confidence\n"
        "- NEVER introduce outside information\n\n"
        
        "## Understanding NARS Statements:\n"
        "- Truth values range from DEFINITELY TRUE to DEFINITELY FALSE\n"
        "- Confidence levels range from EXTREMELY CONFIDENT to EXTREMELY UNCERTAIN\n"
        "- 'X is Y' means 'X is a Y'\n"
        "- 'X is it leads to Y is it' means 'If X then Y'\n\n"
        
        "Remember: All reasoning must be based EXCLUSIVELY on the provided knowledge.\n"
        "Your answer must have a well-defined relationship to your reasoning.\n"
        "---------------------------\n"
        "Knowledge base:\n{context}\n"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])