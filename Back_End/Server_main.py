from flask import Flask,request,jsonify,render_template
from flask_cors import CORS
from model import Account

from Predict import Predict


app = Flask(__name__)

CORS(app,supports_credentials = True)

max_sp_dress = 105
max_dress = 251
max_roles = 65
max_retinue = 22
max_sp_goods = 39
max_goods = 92
max_head = 35
max_head_frame = 33
max_garrtifi = 99
max_waiting_action = 56
max_pursue_action = 60
max_music = 30
max_score = 314000
max_price = 64000
        #self.
# account_item = [
#     {
#         'ssp_dress':1,
#         'sp_dress':2,
#         'dress':3,
#         'roles':5,
#         'retinue':123,
#         'ssp_goods':15,
#         'sp_goods':17,
#         'goods':18,
#         'head':123,
#         'head_frame':34,
#         'graffiti':123,
#         'waiting_action':5,
#         'pursue_action':34,
#         'music':34,
#         'score':3455,
#         'platform':0,
#         'price':2000,
#     }
# ]

@app.route('/hello')
def hello():
    return "hello"

@app.route('/',methods = ['GET','POST'])
def say_hello():
    response_object = {'status':'success'}
    if request.method == 'POST':
        print(request.get_json())
    return response_object

# def item_form():
#     response_object = {'status': 'success'}
#     if request.method == 'GET':
#         getdata = request.get_json()
#         ssp_dress = getdata.get('ssp_dress')
#         sp_dress = getdata.get('sp_dress')/max_sp_dress
#         dress = getdata.get('dress')/max_dress
#         roles = getdata.get('roles')/max_roles
#         retinue = getdata.get('retinue')/max_retinue
#         ssp_goods = getdata.get('ssp_goods')
#         sp_goods = getdata.get('sp_goods')/max_sp_goods
#         goods = getdata.get('goods')/max_goods
#         head = getdata.get('head')/max_head
#         head_frame = getdata.get('head_frame')/max_head_frame
#         graffiti = getdata.get('graffiti')/max_garrtifi
#         waiting_action = getdata.get('waiting_action')/max_waiting_action
#         purse_action = getdata.get('pursue_action')/max_pursue_action
#         music = getdata.get('music')/max_music
#         score = getdata.get('score')/max_score
#         platform = getdata.get('platform')
#         price = getdata.get('price')/max_price
#         if platform == '藏宝阁':
#             platform = 1
#         elif platform == '交易猫':
#             platform = 0
#         else:
#             response_object = {'status':'failed'}
#
#         account_item = [ssp_dress,sp_dress,dress,roles,retinue,ssp_goods,sp_goods,goods,head,head_frame,graffiti,waiting_action,purse_action,music,score,platform,price]
#         account = Account(account_item)
#         pre = account.get_predict()
#         response_object['pre'] = pre
#         return jsonify(response_object)


if __name__ == '__main__':
    app.run()