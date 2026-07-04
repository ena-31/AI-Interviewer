# graph.py
from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv(".env.local")

# -------------------- Build your Interview RAG pipeline --------------------
def create_workflow():
    llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2
    )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "Pinwheel_Robotics_Company_Profile.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}. Please set COMPANY_PDF_PATH environment variable or place TechCompanyInfo.pdf in the current directory.")

    pages = PyPDFLoader(pdf_path).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages_split = text_splitter.split_documents(pages)

    persist_directory = os.getenv("CHROMA_DIR", "./chroma_store")
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="company_info",
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    @tool
    def company_info_tool(query: str) -> str:
        """Searches the company information document and returns relevant chunks about the company."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the company documents."
        # Build the result string step by step for beginners
        result_parts = []
        for i, doc in enumerate(docs):
            info_number = i + 1
            content = doc.page_content
            formatted_info = f"Info {info_number}:\n{content}"
            result_parts.append(formatted_info)
        
        # Join all parts with double newlines
        return "\n\n".join(result_parts)

    @tool
    def record_answer_tool(answer: str) -> str:
        """Records the candidate's answer to a text file for later review."""
        # Simply write the answer to the file
        with open("interview_answers.txt", "a", encoding="utf-8") as f:
            f.write(f"\nAnswer:\n{answer}\n")
            f.write("-" * 50 + "\n")
        
        print(f"Recorded answer: {answer[:50]}...")
        return "Answer recorded successfully!"

    tools = [company_info_tool, record_answer_tool]
    llm = llm.bind_tools(tools)

    class InterviewState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    class STARRRating(BaseModel):
        Situation: int = Field(description="1-10 rating of the situation described by the candidate")
        Task: int = Field(description="1-10 rating of the task described by the candidate")
        Action: int = Field(description="1-10 rating of the action described by the candidate")
        Result: int = Field(description="1-10 rating of the result described by the candidate")
        Reflection: int = Field(description="1-10 rating of the reflection described by the candidate")
        Feedback: str = Field(description="Constructive feedback based on the candidate's responses")

    def get_starr_rating(state: InterviewState) -> InterviewState: 
        """Get a STARR rating based on the candidate's responses."""
        scorer = llm.with_structured_output(STARRRating)
        starr_rating = scorer.invoke(state["messages"])

        text = (
        f"STARR Breakdown:\n"
        f"Situation: {starr_rating.Situation}/10\n"
        f"Task: {starr_rating.Task}/10\n"
        f"Action: {starr_rating.Action}/10\n"
        f"Result: {starr_rating.Result}/10\n\n"
        f"Reflection: {starr_rating.Reflection}/10\n"
        f"Feedback: {starr_rating.Feedback}"
    )

        message = AIMessage(content=text)
        return {"messages": [message]}


    def decide_next_action(state: InterviewState) -> str:
        """Decide what to do next: tool_executor or end"""
        last = state["messages"][-1]
        
        # Check if we need to execute tool calls from the LLM
        if hasattr(last, "tool_calls") and last.tool_calls and len(last.tool_calls) > 0:
            return "tool_executor"
        
        content = last.content if isinstance(last.content, str) else last.content[0].get("text", "")
        if "interview completed" in content.lower():
             return "starr"
        
        
        # Default to end if no specific action needed
        return "end"

    def call_llm(state: InterviewState) -> InterviewState:
        """Main LLM call that handles the interview conversation."""
        system_prompt = (
            "You are a professional interviewer conducting a job interview. By the end of the interview if there is no more question left you will say 'Interview completed I will now provide a STARR rating based on the candidate's responses.' and then go to starr"
            "You will ask structured questions in this order:\n"
            "1. First: 'Hello! Thank you for joining us today. Let's start with the basics - could you tell me about yourself? Please share your background, what you're passionate about, and what brings you here today.'\n"
            "2. After they respond: 'That's great to hear! Now, I'd love to learn about your technical background. Could you tell me about your experience with technology? What technologies, programming languages, or technical projects have you worked with?'\n"
            "3. After they respond: 'Excellent! Now, I'd like to hear about a time when you faced a significant challenge, either technical or professional. Could you walk me through the situation, what obstacles you encountered, and how you overcame them? What did you learn from that experience?'\n"
            "4. After they respond: 'Thank you for sharing that with me. Now, I'd like to give you the opportunity to ask me anything about our company, the role, or anything else you'd like to know. What questions do you have for me?'\n\n"
            "IMPORTANT ROUTING RULES:\n"
            "- When the candidate asks questions about the company (mission, culture, revenue, etc.), use the company_info_tool to find relevant information\n"
            "- When the candidate gives answers to your interview questions, use the record_answer_tool to record their response, then acknowledge it and ask the next question\n"
            "- Be conversational, professional, and helpful\n"
            "- If you don't have specific company information, say so honestly and offer to connect them with someone who might know more"
        )
        
        msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
        message = llm.invoke(msgs)
        return {"messages": [message]}

    def tool_executor(state: InterviewState) -> InterviewState:
        """Execute tool calls from the LLM's response."""
        # Get the tool calls from the last message
        tool_calls = state["messages"][-1].tool_calls
        results = []

        # Go through each tool call one by one
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            
            print(f"Running tool: {tool_name}")

            # Call the right tool based on its name
            if tool_name == "company_info_tool":
                result = company_info_tool.invoke(tool_args)
            elif tool_name == "record_answer_tool":
                result = record_answer_tool.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"

            # Create a message with the result
            tool_message = ToolMessage(
                tool_call_id=tool_call["id"],
                name=tool_name,
                content=str(result),
            )
            results.append(tool_message)

        print("All tools finished running.")
        return {"messages": results}


    # Build the interview graph
    graph = StateGraph(InterviewState)
    
    # Add nodes
    graph.add_node("llm", call_llm)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("starr", get_starr_rating)
    
    # Set up the flow
    graph.set_entry_point("llm")
    
    # Conditional edges from LLM
    graph.add_conditional_edges(
        "llm", 
        decide_next_action, 
        {
            "tool_executor": "tool_executor",
            "starr": "starr"
        }
    )
    
    # From tool_executor back to LLM
    graph.add_edge("tool_executor", "llm")
    graph.add_edge("starr", END)

    return graph.compile()