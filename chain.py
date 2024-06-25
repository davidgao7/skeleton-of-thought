from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


skeleton_generator_template = """
[User:] You’re an organizer responsible for only giving the skeleton (not the full content) 
for answering the question. Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) 
to answer the question. Instead of writing a full sentence, each skeleton point should be very 
short with only 3∼5 words. Generally, the skeleton should have 3∼10 points. 
Now, please provide the skeleton for the following question. {question} Skeleton: [Assistant:] 1.
"""

skeleton_generator_prompt = ChatPromptTemplate.from_template(
    skeleton_generator_template
)

skeleton_generator_chain = skeleton_generator_prompt | ChatOpenAI() | StrOutputParser()

point_expander_template = """
[User:] You’re responsible for continuing the writing of one and only one point in the overall 
answer to the following question. 

{question} 

The skeleton of the answer is 
{skeleton}

Continue and only continue the writing of point {point_index}. Write it **very shortly** in 1∼2 
sentence and do not continue with other points! 

[Assistant:] {point_index}. {point_skeleton}
"""

point_expander_prompt = ChatPromptTemplate.from_template(point_expander_template)

point_expander_chain = point_expander_prompt | ChatOpenAI() | StrOutputParser()

chain = RunnablePassthrough.assign(skeleton=skeleton_generator_chain)


if __name__ == "__main__":

    # base output from gpt
    skeleton = """
        1. Open communication 
        2. Active listening 
        3. Collaboration 
        4. Mediation 
        5. Conflict resolution training 
        6. Establishing clear policies 
        7. Addressing issues promptly
    """

    point_expander_chain.invoke(
        {
            "question": "What are the most effective strategies for conflict resolution in the workplace?",
            "skeleton": skeleton,
            "point_index": 1,
            "point_skeleton": "Open communication and active listening.",
        }
    )
