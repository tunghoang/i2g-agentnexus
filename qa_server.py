import uuid
import traceback
import json
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
__APP_VERSION__="1.0.0"
# Define request model
class Question(BaseModel):
    agentid: Optional[str]
    question: str

# Define response model
class Answer(BaseModel):
    answer: str

class QAServer(FastAPI):
    def __init__(self, create_agent_fn):
        super().__init__()
        self.agents = dict()
        self.create_agent_fn = create_agent_fn


def clean_response(response: str) -> str:
    """Clean up double-wrapped JSON responses"""
    if isinstance(response, str):
        # Check if it's a JSON response with content array
        if response.startswith('{"content":'):
            try:
                parsed = json.loads(response)
                if "content" in parsed and isinstance(parsed["content"], list):
                    if len(parsed["content"]) > 0 and "text" in parsed["content"][0]:
                        inner_text = parsed["content"][0]["text"]

                        # Check if inner_text is JSON with escaped unicode
                        if inner_text.startswith('{"text":'):
                            inner_parsed = json.loads(inner_text)
                            if "text" in inner_parsed:
                                # Decode unicode escapes
                                clean_text = inner_parsed["text"].encode().decode('unicode_escape')
                                return clean_text

                        return inner_text
                return response
            except json.JSONDecodeError:
                return response
    return response

def qa_server_create(create_agent_fn):
    app = QAServer(create_agent_fn)

    def get_app() -> QAServer:
        return app

    @app.post("/ask", response_model=Answer)
    def ask_question(question: Question, qa_server: QAServer = Depends(get_app)):
        agentids = list([k for k in qa_server.agents if qa_server.agents[k]['killed'] == 0])
        if len(agentids) == 0:
            raise HTTPException(status_code=404, detail="No agents exist")
        agentid = question.agentid or agentids[0]
        agentData = qa_server.agents.get(f"{agentid}", None)
        if agentData is None:
            raise HTTPException(status_code=404, detail=f"agent {agentid} not found")
        if agentData['killed'] == 1:
            raise HTTPException(status_code=404, detail=f"agent {agentid} was killed")
        agentData['tl'] = datetime.now()
        agent = agentData['agent']

        # process query
        user_input = question.question
        if not user_input:
            return Answer(answer="Your question is empty")

        command = user_input.lower().strip()

        if command in [ 'version', 'ver' ]:
            return Answer(answer=__APP_VERSION__)
        try:
            response = agent.run(user_input)
            response = clean_response(response)
        except:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Server Error")
        return Answer(answer=response)

    @app.get("/agents")
    def list_agents(qa_server: QAServer = Depends(get_app)):
        agentlist = [k for k in qa_server.agents if qa_server.agents[k]['killed'] == 0]
        return list(agentlist)

    @app.get("/new")
    def create_agent(qa_server: QAServer = Depends(get_app)):
        agentid = str(uuid.uuid4())
        new_agent = qa_server.create_agent_fn()
        now = datetime.now()
        qa_server.agents[agentid] = dict(t0=now, tl=now, agent=new_agent, killed=0)
        return dict(agentid=agentid)
    return app

