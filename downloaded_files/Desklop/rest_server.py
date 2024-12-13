#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or 18.04
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
REST-сервер для взаимодействия с ботом. Используется Flask и WSGIServer.
'''

import os
import sys
import signal
import platform
import base64
import json
import subprocess
import socket
from logging.config import dictConfig
from datetime import datetime
from functools import wraps

from flask import Flask, redirect, jsonify, abort, request, make_response, __version__ as flask_version
from flask_httpauth import HTTPBasicAuth
from gevent import __version__ as wsgi_version
from gevent.pywsgi import WSGIServer

from tensorflow import get_default_graph

from text_to_text import TextToText
from text_to_speech import TextToSpeech
from speech_to_text import SpeechToText
from source_to_prepared import SourceToPrepared

# Создание временной папки, если она была удалена
if not os.path.exists('temp'):
    os.makedirs('temp')

# Создание папки для логов, если она была удалена
if not os.path.exists('log'):
    os.makedirs('log')

# Удаление старых логов
if os.path.exists('log/server.log'):
    os.remove('log/server.log')
    for i in range(1, 6):
        if os.path.exists('log/server.log.' + str(i)):
            os.remove('log/server.log.' + str(i))

# Конфигурация логгера
dictConfig({
    'version' : 1,
    'formatters' : {
        'simple' : {
            'format' : '%(levelname)-8s | %(message)s'
        }
    },
    'handlers' : {
        'console' : {
            'class' : 'logging.StreamHandler',
            'level' : 'DEBUG',
            'formatter' : 'simple',
            'stream' : 'ext://sys.stdout'
        },
        'file' : {
            'class' : 'logging.handlers.RotatingFileHandler',
            'level' : 'DEBUG',
            'maxBytes' : 16 * 1024 * 1024,
            'backupCount' : 5,
            'formatter' : 'simple',
            'filename' : 'log/server.log'
        }
    },
    'loggers' : {
        'console' : {
            'level' : 'DEBUG',
            'handlers' : ['console'],
            'propagate' : 'no'
        },
        'file' : {
            'level' : 'DEBUG',
            'handlers' : ['file'],
            'propagate' : 'no'
        }
    },
    'root' : {
        'level' : 'DEBUG',
        'handlers' : ['console', 'file']
    }
})

app = Flask(__name__)
auth = HTTPBasicAuth()
max_content_length = 16 * 1024 * 1024
f_name_plays = 'data/plays_ru/plays_ru.txt'
f_name_w2v_model_plays = 'data/plays_ru/w2v_model_plays_ru.bin'
f_name_model_plays = 'data/plays_ru/model_plays_ru.json'
f_name_model_weights_plays = 'data/plays_ru/model_weights_plays_ru.h5'
f_name_audio = 'temp/synthesized_speech.wav'


def limit_content_length():
    ''' Декоратор для ограничения размера передаваемых клиентом данных. '''
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if request.content_length > max_content_length:            
                log('превышен максимальный размер передаваемых данных ({:.2f} кБ)'.format(request.content_length/1024), request.remote_addr, 'error')
                return make_response(jsonify({'error': 'Maximum data transfer size exceeded, allowed only until {: .2f} kB.'.format(max_content_length/1024)}), 413)
            elif request.content_length == 0:
                log('тело запроса не содержит данных', request.remote_addr, 'error')
                return make_response(jsonify({'error': 'The request body contains no data.'}), 400)
            elif request.json is None:
                log('тело запроса содержит неподдерживаемый тип данных', request.remote_addr, 'error')
                return make_response(jsonify({'error': 'The request body contains an unsupported data type. Only json is supported'}), 415)
            return f(*args, **kwargs)
        return wrapper
    return decorator


def log(message, addr=None, level='info'):
    ''' Запись сообщения в лог файл с уровнем INFO или ERROR. По умолчанию используется INFO.
    1. addr - строка с адресом подключённого клиента
    2. message - сообщение
    3. level - уровень логгирования, может иметь значение либо 'info', либо 'error' '''
    if level == 'info':
        if addr is None:
            app.logger.info(datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)
        else:
            app.logger.info(addr + ' - - ' + datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)
    elif level == 'error':
        if addr is None:
            app.logger.error(datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)
        else:
            app.logger.error(addr + ' - - ' + datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)

ttt = None
stt = None
tts = None
http_server = None
# Получение графа вычислений tensorflow по умолчанию (для последующей передачи в другой поток)
graph = get_default_graph()


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'The requested URL was not found on the server.'}), 404)


@app.errorhandler(405)
def method_not_allowed(error):
    return make_response(jsonify({'error': 'The method is not allowed for the requested URL.'}), 405)


@app.errorhandler(500)
def internal_server_error(error):
    print(error)
    return make_response(jsonify({'error': 'The server encountered an internal error and was unable to complete your request.'}), 500)


@auth.get_password
def get_password(username):
    # login bot, password test_bot
    if username == 'bot':
        return 'test_bot'


@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access.'}), 401)


@app.route('/', methods=['GET'])
@app.route('/chatbot', methods=['GET'])
def root():
    return redirect('/chatbot/about')


@app.route('/chatbot/about', methods=['GET'])
@auth.login_required
def about():
    ''' Возвращает информацию о проекте. '''
    return jsonify({'text':'Чат-бот на основе нейронной сети AttentionSeq2Seq. Обучен на диалогах из пьес. Поддерживает общение ' + \
                           'в текстовом формате, с синтезом (RHVoice) и распознаванием (PocketSphinx) речи. Подробнее в ' + \
                           'https://github.com/Desklop/Voice_ChatBot'})


@app.route('/chatbot/questions', methods=['GET'])
@auth.login_required
def questions():
    ''' Возвращает список всех поддерживаемых ботом вопросов. '''
    stp = SourceToPrepared()
    questions = stp.get_questions(f_name_plays)
    return jsonify({'text':questions})


@app.route('/chatbot/speech-to-text', methods=['POST'])
@auth.login_required
@limit_content_length()
def speech_to_text():
    ''' Принимает .wav/.opus файл с записанной речью, распознаёт её с помощью PocketSphinx и возвращает распознанную строку. '''    
    data = request.json
    audio = data.get('wav')
    audio_format = 'wav'
    if audio is None:
        audio = data.get('opus')
        audio_format = 'opus'
        if audio is None:
            log('json в теле запроса имеет неправильную структуру', request.remote_addr, 'error')
            return make_response(jsonify({'error': 'Json in the request body has an invalid structure.'}), 415)
    audio = base64.b64decode(audio)
    with open('temp/speech.' + audio_format, 'wb') as audiofile:
        audiofile.write(audio)
    log('принят .{} размером {:.2f} кБ, сохранено в temp/speech.{}'.format(audio_format, len(audio)/1024, audio_format), request.remote_addr)  

    question = stt.get('temp/speech.' + audio_format) # Первый раз распознаёт не очень, т.к. параллельно подстраиваются фильтры и т.д
    question = stt.get('temp/speech.' + audio_format) # Когда второй раз одну и ту же фразу - распознавание куда лучше

    if question == 'error':
        log('json в теле запроса содержит некорректные данные', request.remote_addr, 'error')
        return make_response(jsonify({'error': 'Json in the request body contains incorrect data.'}), 415)
    log("распознано: '" + question + "'", request.remote_addr)
    return jsonify({'text':question})


@app.route('/chatbot/text-to-speech', methods=['POST'])
@auth.login_required
@limit_content_length()
def text_to_speech():
    ''' Принимает строку, синтезирует речь с помощью RHVoice и возвращает .wav файл с синтезированной речью. '''    
    data = request.json
    data = data.get('text')
    if data is None:
        log('json в теле запроса имеет неправильную структуру', request.remote_addr, 'error')
        return make_response(jsonify({'error': 'Json in the request body has an invalid structure.'}), 415)
    log("принято: '" + data + "'", request.remote_addr)

    tts.get(data, f_name_audio)

    with open(f_name_audio, 'rb') as audiofile:
        audio = audiofile.read()    
    log('создан .wav размером {:.2f} кБ, сохранено в {}'.format(len(audio)/1024, f_name_audio), request.remote_addr)
    audio = base64.b64encode(audio)
    return jsonify({'wav':audio.decode()})


@app.route('/chatbot/text-to-text', methods=['POST'])
@auth.login_required
@limit_content_length()
def text_to_text():
    ''' Принимает строку с вопросом к боту и возвращает ответ в виде строки. '''
    data = request.json
    data = data.get('text')
    if data is None:
        log('json в теле запроса имеет неправильную структуру', request.remote_addr, 'error')
        return make_response(jsonify({'error': 'Json in the request body has an invalid structure.'}), 415)
    log("принято: '" + data + "'", request.remote_addr)

    with graph.as_default():
        answer = ttt.predict(data)

    log("ответ: '" + answer + "'", request.remote_addr)
    return jsonify({'text':answer})

# Всего 5 запросов:
# 1. GET-запрос на /chatbot/about, вернёт инфу о проекте
# 2. GET-запрос на /chatbot/questions, вернёт список всех вопросов
# 3. POST-запрос на /chatbot/speech-to-text, принимает .wav/.opus-файл и возвращает распознанную строку
# 4. POST-запрос на /chatbot/text-to-speech, принимает строку и возвращает .wav-файл с синтезированной речью
# 5. POST-запрос на /chatbot/text-to-text, принимает строку и возвращает ответ бота в виде строки

def run(host, port, wsgi=False, https_mode=False):
    ''' Автовыбор доступного порта (если указан порт 0), загрузка языковой модели и нейронной сети и запуск сервера.
    1. wsgi - True: запуск WSGI сервера, False: запуск тестового Flask сервера
    2. https - True: запуск в режиме https (сертификат и ключ должны быть в cert.pem и key.pem), False: запуск в режиме http
    
    Самоподписанный сертификат можно получить, выполнив: openssl req -x509 -newkey rsa:4096 -nodes -out temp/cert.pem -keyout temp/key.pem -days 365 '''
    
    if port == 0: # Если был введён порт 0, то автовыбор любого доступного порта
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, 0))
            port = sock.getsockname()[1]
            log('выбран порт ' + str(port))
            sock.close()
        except socket.gaierror:
            log('адрес ' + host + ':' + str(port) + ' некорректен', level='error')
            sock.close()
            return
        except OSError:
            log('адрес ' + host + ':' + str(port) + ' недоступен', level='error')
            sock.close()
            return

    log('Flask v.' + flask_version + ', WSGIServer v.' + wsgi_version)
    log('установлен максимальный размер принимаемых данных: {:.2f} Кб'.format(max_content_length/1024))
    
    name_dataset = f_name_w2v_model_plays[f_name_w2v_model_plays.rfind('w2v_model_')+len('w2v_model_'):f_name_w2v_model_plays.rfind('.bin')]
    log('загрузка обученной на наборе данных ' + name_dataset + ' модели seq2seq...')
    global ttt
    print()
    ttt = TextToText(f_name_w2v_model=f_name_w2v_model_plays, f_name_model=f_name_model_plays, f_name_model_weights=f_name_model_weights_plays)
    print()

    log('загрузка языковой модели для распознавания речи...')
    global stt
    stt = SpeechToText('from_file', name_dataset)

    log('загрузка синтезатора речи...')
    global tts
    tts = TextToSpeech('anna')

    if wsgi:
        global http_server
        if https_mode:
            log('WSGI сервер запущен на https://' + host + ':' + str(port) + ' (нажмите Ctrl+C или Ctrl+Z для выхода)')
        else:
            log('WSGI сервер запущен на http://' + host + ':' + str(port) + ' (нажмите Ctrl+C или Ctrl+Z для выхода)')
        try:
            if https_mode:
                http_server = WSGIServer((host, port), app, log=app.logger, error_log=app.logger, keyfile='temp/key.pem', certfile='temp/cert.pem')
            else:
                http_server = WSGIServer((host, port), app, log=app.logger, error_log=app.logger)
            http_server.serve_forever()
        except OSError:
            print()
            log('адрес ' + host + ':' + str(port) + ' недоступен', level='error')
    else:
        log('запуск тестового Flask сервера...')
        try:
            if https_mode:
                app.run(host=host, port=port, ssl_context=('temp/cert.pem', 'temp/key.pem'), threaded=True, debug=False)
            else:
                app.run(host=host, port=port, threaded=True, debug=False)
        except OSError:
            print()
            log('адрес ' + host + ':' + str(port) + ' недоступен', level='error')


def get_address_on_local_network():
    ''' Определение адреса машины в локальной сети с помощью утилиты 'ifconfig' из пакета net-tools.
    1. возвращает строку с адресом или 127.0.0.1, если локальный адрес начинается не с 192.Х.Х.Х или 172.Х.Х.Х
    
    Проверено в Ubuntu 16.04 и 18.04. '''
    
    command_line = 'ifconfig'
    proc = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    out = out.decode()
    if out.find('not found') != -1:
        print("\n[E] 'ifconfig' не найден.")
        sys.exit(0)
    if out.find('inet 127.0.0.1') != -1:
        template = 'inet '
    elif out.find('inet addr:127.0.0.1') != -1:
        template = 'inet addr:'
    i = 0
    host_192xxx = None
    host_172xxx = None
    while host_192xxx is None or host_172xxx is None: 
        out = out[out.find(template) + len(template):]
        host = out[:out.find(' ')]
        out = out[out.find(' '):]
        if host.find('192.168') != -1:
            host_192xxx = host
        elif host.find('172.') != -1:
            host_172xxx = host
        i += 1
        if i >= 10:
            break
    if host_192xxx:
        return host_192xxx
    elif host_172xxx:
        return host_172xxx
    else:
        print('\n[E] Неподдерживаемый формат локального адреса, требуется корректировка исходного кода.\n')
        return '127.0.0.1'

# Добавить админку (которая будет отдавать простую html, в которой можно тестировать все запросы)
# Добавить запись необработанных слов в отдельный файл и конфиг-файл
# Попробовать распознавание речи с помощью Kaldi и MozillaDeepSpeech

def main():
    host = '127.0.0.1'
    port = 5000
    
    # Аргументы командной строки имеют следующую структуру: [ключ(-и)] [адрес:порт]
    # rest_server.py - запуск WSGI сервера с автоопределением адреса машины в локальной сети и портом 5000
    # rest_server.py host:port - запуск WSGI сервера на host:port
    # rest_server.py -d - запуск тестового Flask сервера на 127.0.0.1:5000
    # rest_server.py -d host:port - запуск тестового Flask сервера на host:port    
    # rest_server.py -d localaddr:port - запуск тестового Flask сервера с автоопределением адреса машины в локальной сети и портом port
    # rest_server.py -s - запуск WSGI сервера с поддержкой https, автоопределением адреса машины в локальной сети и портом 5000
    # rest_server.py -s host:port - запуск WSGI сервера c поддержкой https на host:port
    # rest_server.py -s -d - запуск тестового Flask сервера c поддержкой https на 127.0.0.1:5000
    # rest_server.py -s -d host:port - запуск тестового Flask сервера c поддержкой https на host:port    
    # rest_server.py -s -d localaddr:port - запуск тестового Flask сервера c поддержкой https, автоопределением адреса машины в локальной сети и портом port
    # Что бы выбрать доступный порт автоматически, укажите в host:port или localaddr:port порт 0

    #run(host, port, wsgi=False)

    if len(sys.argv) > 1:
        if sys.argv[1] == '-s': # запуск в режиме https
            if len(sys.argv) > 2:
                if sys.argv[2] == '-d': # запуск тестового Flask сервера
                    if len(sys.argv) > 3:
                        if sys.argv[3].find('localaddr') != -1 and sys.argv[3].find(':') != -1: # localaddr:port
                            host = get_address_on_local_network()
                            port = int(sys.argv[3][sys.argv[3].find(':') + 1:])
                            run(host, port, https_mode=True)
                        elif sys.argv[3].count('.') == 3 and sys.argv[3].count(':') == 1: # host:port         
                            host = sys.argv[3][:sys.argv[3].find(':')]
                            port = int(sys.argv[3][sys.argv[3].find(':') + 1:])
                            run(host, port, https_mode=True)
                        else:
                            print("\n[E] Неверный аргумент командной строки '" + sys.argv[3] + "'. Введите help для помощи.\n")
                    else:
                        run(host, port, https_mode=True)

                elif sys.argv[2].count('.') == 3 and sys.argv[2].count(':') == 1: # запуск WSGI сервера на host:port
                    host = sys.argv[2][:sys.argv[2].find(':')]
                    port = int(sys.argv[2][sys.argv[2].find(':') + 1:])
                    run(host, port, wsgi=True, https_mode=True)

                else:
                    print("\n[E] Неверный аргумент командной строки '" + sys.argv[2] + "'. Введите help для помощи.\n")
            else: 
                host = get_address_on_local_network()
                run(host, port, wsgi=True, https_mode=True)

        elif sys.argv[1] == '-d': # запуск тестового Flask сервера
            if len(sys.argv) > 2:
                if sys.argv[2].find('localaddr') != -1 and sys.argv[2].find(':') != -1: # localaddr:port
                    host = get_address_on_local_network()
                    port = int(sys.argv[2][sys.argv[2].find(':') + 1:])
                    run(host, port)
                elif sys.argv[2].count('.') == 3 and sys.argv[2].count(':') == 1: # host:port
                    host = sys.argv[2][:sys.argv[2].find(':')]
                    port = int(sys.argv[2][sys.argv[2].find(':') + 1:])
                    run(host, port)                
                else:
                    print("\n[E] Неверный аргумент командной строки '" + sys.argv[2] + "'. Введите help для помощи.\n")
            else:
                run(host, port)

        elif sys.argv[1].count('.') == 3 and sys.argv[1].count(':') == 1: # запуск WSGI сервера на host:port
            host = sys.argv[1][:sys.argv[1].find(':')]
            port = int(sys.argv[1][sys.argv[1].find(':') + 1:])
            run(host, port, wsgi=True)
        elif sys.argv[1] == 'help':
            print('\nПоддерживаемые варианты работы:')
            print('\tбез аргументов - запуск WSGI сервера с автоопределением адреса машины в локальной сети и портом 5000')
            print('\thost:port - запуск WSGI сервера на host:port')
            print('\t-d - запуск тестового Flask сервера на 127.0.0.1:5000')
            print('\t-d host:port - запуск тестового Flask сервера на host:port')
            print('\t-d localaddr:port - запуск тестового Flask сервера с автоопределением адреса машины в локальной сети и портом port')
            print('\t-s - запуск WSGI сервера с поддержкой https, автоопределением адреса машины в локальной сети и портом 5000')
            print('\t-s host:port - запуск WSGI сервера с поддержкой https на host:port')
            print('\t-s -d - запуск тестового Flask сервера с поддержкой https на 127.0.0.1:5000')
            print('\t-s -d host:port - запуск тестового Flask сервера с поддержкой https на host:port')
            print('\t-s -d localaddr:port - запуск тестового Flask сервера с поддержкой https, автоопределением адреса машины в локальной сети и портом port\n')
        else:
            print("\n[E] Неверный аргумент командной строки '" + sys.argv[1] + "'. Введите help для помощи.\n")
    else: # запуск WSGI сервера с автоопределением адреса машины в локальной сети и портом 5000
        host = get_address_on_local_network()
        run(host, port, wsgi=True)


def on_stop(*args):
    print()
    log('сервер остановлен')
    if http_server != None:
        http_server.close()
    sys.exit(0)


if __name__ == '__main__':
    # При нажатии комбинаций Ctrl+Z, Ctrl+C либо закрытии терминала будет вызываться функция on_stop() (Работает только на linux системах!)
    if platform.system() == 'Linux':
        for sig in (signal.SIGTSTP, signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, on_stop)
    main()
    
'''
if sys.argv[1].count('*') > 0: # для задания максимальной длины принимаемых данных в виде выражения 
    print('это выражение!')
    temp = sys.argv[1]
    new_content_length = 1
    while temp.count('*') > 0:
        new_content_length *= int(temp[:temp.find('*')])
        temp = temp[temp.find('*') + 1:]
    new_content_length *= int(temp)
    max_content_length = new_content_length
    print(max_content_length)
elif sys.argv[1].isdigit() == True: # для задания максимальной длины принимаемых данных в виде одного числа 
    print('это просто число!')
    print(int(sys.argv[1]))
    max_content_length = int(sys.argv[1]) * 1024
    print(max_content_length)
'''