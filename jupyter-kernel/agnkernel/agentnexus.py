from ipykernel.kernelbase import Kernel
import requests
import os, logging
import sys
import re
import json

QA_SERVER_BASE_URL = 'http://localhost:8990'

class REPLKernel(Kernel):
    implementation = 'agentnexus'
    implementation_version = '1.0'
    banner = "GenAI for Subsurface domain analysis"
    language = 'no-op'
    language_version = '1.0'

    language_info = {'name': 'agentnexus',
            'mime_type': 'text/plain',
            'file_extension': '.txt'}

    def __init__(self, **kwargs):
        Kernel.__init__(self, **kwargs)
        self._start_app()
    def _start_app(self):
        response = requests.get(f'{QA_SERVER_BASE_URL}/agents')
        if response.status_code == 200:
            resJson = response.json()
            if isinstance(resJson, list) and len(resJson) > 0:
                self.agentid = resJson[0]
            else:
                response = requests.get(f'{QA_SERVER_BASE_URL}/new')
                if response.status_code == 200:
                    resJson = response.json()
                    self.agentid = resJson['agentid']
                else:
                    raise Exception('Error in creating agent')
        else:
            raise Exception('Error contacting QA server')


    def process_output(self, output, mimetype='text/markdown'):
        #self.send_response(self.iopub_socket, 'stream', {'name': 'stdout', 'text': f"{output}"})
        self.send_response(self.iopub_socket, 'execute_result', {
            'execution_count': 1,
            'data': {
                mimetype: f"{output}",
            }, 
            'metadata': {}
        })

    def do_execute(self, code, silent, store_history = True, user_expressions=None, allow_stdin = False):
        response = requests.post(f'{QA_SERVER_BASE_URL}/ask', json=dict(agentid=self.agentid, question=code))
        if response.status_code == 200:
            resJson = response.json()
            answer = resJson['answer']
            mimetype = resJson['mimetype']
            self.process_output(answer, mimetype=mimetype)
        elif response.status_code == 411:
            response1 = requests.get(f'{QA_SERVER_BASE_URL}/new')
            if response1.status_code == 200:
                resJson = response1.json()
                self.agentid = resJson['agentid']
                self.process_output(f'{response.text}. New agent created. Please ask again!')
            else:
                self.process_output('Error in creating agent')
        else:
            self.process_output(response.text)
        return {'status': 'ok', 'execution_count': self.execution_count,
                    'payload': [], 'user_expressions': {}}
