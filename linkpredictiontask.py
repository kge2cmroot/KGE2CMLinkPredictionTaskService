
import numpy as np
import pandas as pd
import ampligraph
import os
import threading

import json
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


class KnoledgeDiscoveryResult:
  def __init__(self, score_result, head, relation, tail):
    self.score_result = score_result
    self.head = head
    self.relation = relation
    self.tail=tail
  
  def jsonR (self):
    return {'head':self.head, 'relation':self.relation, 'tail':self.tail}

  
 

class KnoledgeDiscoveryResultEncoder(json.JSONEncoder):
    def default(self, obj):
            return {'head':obj.head, 'relation':obj.relation, 'tail':obj.tail}

def transformResult(_triple, _score):
    return KnoledgeDiscoveryResult(float(_score), _triple[0], _triple[1], _triple[2])

def transformResultToJson(_result):
    return json.dumps(_result, cls=KnoledgeDiscoveryResultEncoder)
    
def t(_knoledgeDiscoveryResult):
    return [_knoledgeDiscoveryResult.head, _knoledgeDiscoveryResult.relation, _knoledgeDiscoveryResult.tail]

from ampligraph.latent_features import restore_model
from ampligraph.discovery import query_topn

model = restore_model('./science-model.pkl')

def eval(_head, _tail):
  triples=[]
  scores=[]

  try:
    triples, scores = query_topn(model, top_n=10,
                head=_head, relation=None, tail=_tail,
                ents_to_consider=None, rels_to_consider=None)
  except ValueError as err: 
    print(err) 

  # transformed =  map(transformResult, triples, scores)
  # return map(t, transformed)
  return map(transformResult, triples, scores)


from flask import Flask, render_template, request,jsonify
from pyngrok import ngrok



model = restore_model('./science-model.pkl')

os.environ["FLASK_ENV"] = "development"
app = Flask(__name__)
port = 5000
ngrok.set_auth_token("2IHYrvBVmPzlN5d5a5EIh4a2p33_GYcikNX7UZM5UcwtP3PF")
public_url = ngrok.connect(port).public_url
print(public_url, port)

app.config["BASE_URL"] = public_url


@app.route('/predictions', methods=['POST'])
def predictionsAPI():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        body = request.get_json()
        
        
        evaluated = eval(body['head'], body["tail"])

        result = [ x for x in list(evaluated) if x.relation == body["relation"] ]

        if not result:
          return {'head': body['head'], 'tail': body["tail"] }
        else:
          resp = result[0]
          print(resp)
          return resp.jsonR()
          # return {'head': resp['head'], 'relation':resp['relation'], 'tail': resp["tail"] }  
    else:
        return 'Content-Type not supported!'



t = threading.Thread(target=app.run, kwargs={"use_reloader": False})
t.start()