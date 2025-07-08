import os
import re
import json
import logging
import random
import string
import tempfile
import requests
from datetime import datetime
from openai import AzureOpenAI
from dotenv import load_dotenv
# 環境変数を読み込む
load_dotenv()

import streamlit as st
import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.cosmos import PartitionKey
from azure.cosmos.cosmos_client import CosmosClient
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import HttpResponseError

# 他のスクリプトから関数をインポート
from preparedata import process_file

# envファイルから環境変数を取得
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Azure OpenAI Service の情報を環境変数から取得する
AZURE_OPENAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL")
AZURE_OPENAI_EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_MODEL")
AZURE_OPENAI_CHAT_MAX_TOKENS = int(os.getenv("AZURE_OPENAI_CHAT_MAX_TOKENS", "1000"))

# 環境変数から Azure Cosmos DB の接続文字列とデータベース名を取得する
COSMOS_CONNECTION_STRING = os.getenv("COSMOS_CONNECTION_STRING")
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
COSMOS_CONTAINER_NAME_CHAT = os.getenv("COSMOS_CONTAINER_NAME_CHAT")
COSMOS_CONTAINER_NAME_FEEDBACK = os.getenv("COSMOS_CONTAINER_NAME_FEEDBACK")

image_blob_service_client = BlobServiceClient.from_connection_string(os.getenv("IMAGE_BLOB_STORAGE_CONNECTION_STRING"))
image_blob_container_client = image_blob_service_client.get_container_client(os.getenv("IMAGE_BLOB_CONTAINER_NAME"))

# Cosmos DB クライアントを生成する
cosmos_client = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)
database = cosmos_client.get_database_client(COSMOS_DB_NAME)
database.create_container_if_not_exists(
    id=COSMOS_CONTAINER_NAME_CHAT,
    partition_key=PartitionKey(path=f"/id"),
)
database.create_container_if_not_exists(
    id=COSMOS_CONTAINER_NAME_FEEDBACK,
    partition_key=PartitionKey(path="/id"),
)
container = database.get_container_client(COSMOS_CONTAINER_NAME_CHAT)
container_feedback = database.get_container_client(COSMOS_CONTAINER_NAME_FEEDBACK)  # 追加


# Azure AI Search の情報を環境変数から取得する
AI_SEARCH_ENDPOINT = os.getenv("AI_SEARCH_ENDPOINT")
AI_SEARCH_KEY = os.getenv("AI_SEARCH_KEY")
AI_SEARCH_API_VERSION = os.getenv("AI_SEARCH_API_VERSION", "2023-10-01-Preview")
AI_SEARCH_INDEX_NAME = os.getenv("AI_SEARCH_INDEX_NAME")
AI_SEACH_SEMANTIC = os.getenv("AI_SEACH_SEMANTIC")
top_k_temp = 20  # 検索結果の上位何件を表示するか

# Azure Blob Storage の情報を環境変数から取得する
BLOB_STORAGE_CONNECTION_STRING = os.getenv("BLOB_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

# Blob Storage クライアントを作成
blob_service_client = BlobServiceClient.from_connection_string(BLOB_STORAGE_CONNECTION_STRING)
blob_container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
# コンテナが存在しない場合は作成
try:
    blob_container_client.get_container_properties()
except Exception:
    blob_container_client.create_container()

# OpenAIへのプロンプト設計を行う。
SystemPrompt = """あなたは、会社の従業員がSAPのマニュアルに対する質問をする際に支援する優秀なアシスタントです。
以下の制約を必ず守ってユーザの質問に回答してください。
ハルシネーションは起こさないでください。
魅力的で丁寧な回答をする必要があります。
最初から最後までじっくり読んで回答を作ってください。最高の仕事をしましょう
もし複数のドキュメントに該当する内容があればそれらも参考にして回答してください。
・事業にはAIS、ジェネスタ、エラストマー、エバール樹脂、イソプレンケミカル、エバールフィルム、ポバールフィルムがあります。マニュアルには複数の事業専用ものがあります。
ファイル名の先頭ににどの事業のマニュアルか書かれています。
ユーザーの質問がどの事業のマニュアルを検索しているのかを確認の上回答してください。
参考になりそうなURLがあった場合は提示してください。
ただしその時どの事業部（AIS、ジェネスタ、エラストマー、エバール樹脂、イソプレンケミカル、エバールフィルム、ポバールフィルム）のURLか注意して回答するようにしてください。
ユーザーの質問している事業部と合致するURLを回答するようにしてください。
・ユーザーの質問が不明瞭な場合は、明確化のためにユーザに必ず質問してください。
以下の単語はすべて同じ意味を持つものとして扱ってください。
PO, 購買伝票, Purchase Order, 購買発注伝票
以下の単語はすべて同じ意味を持つものとして扱ってください。
SO, Sales Order, 受注伝票,受注
以下の単語はすべて同じ意味を持つものとして扱ってください。
DO,	Delivery Order,	出荷伝票
以下の単語はすべて同じ意味を持つものとして扱ってください。
IV,	請求書,	請求伝票, 請求, インボイス
以下の単語はすべて同じ意味を持つものとして扱ってください。
it.mds,	Glasswing
以下の単語はすべて同じ意味を持つものとして扱ってください。
WMD, xSuite
以下の単語はすべて同じ意味を持つものとして扱ってください。
SNOW, Service Now, ServiceNow
以下の単語はすべて同じ意味を持つものとして扱ってください。
PIR, 購買情報マスタ

# 制約 
・以下のSources(情報源)に記載されたコンテキストのみを使用して回答してください。必ず情報源に記載されたコンテキストを基に回答を作ってください
・十分な情報がない場合は、わからないと回答してください。
・以下のSources(情報源)を使用しない回答は生成しないでください 。回答には役割(userやassistantなど)の情報を含めないでください。
・ユーザーの質問が不明瞭な場合は、明確化のためにユーザに必ず質問してください。
・Sourcesには、名前の後にコロンと実際の情報が続きます。回答で使用する各事実について、常にSourcesの情報を含めてください。
  情報源を参照するには、各Content情報の前段にあるfilenameの情報を反映してください。角かっこを使用してください。
  Sources参照ルール：[filename] 　Sources出力例：[info1.txt]
・Sourcesを組み合わせないでください。各Sourcesを個別にリストしてください。例：[info1.txt],[info1.txt]
・日本語の質問の場合は、日本語で回答を作成してください。英語での質問の場合は、英語で回答を作成し回答してください。    
"""

# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text, text_limit=7000, compress_to_1024=False):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\n\r]+', ' ', text).strip()
    if len(text) > text_limit:
        logging.warning("Token limit exceeded maximum length, truncating...")
        text = text[:text_limit]

    response = client.embeddings.create(model=AZURE_OPENAI_EMBED_MODEL, input=text)
    embedding = response.data[0].embedding

    if compress_to_1024:
        if len(embedding) == 3072:
            embedding = np.array(embedding).reshape(3, 1024).mean(axis=0).tolist()
        else:
            raise ValueError(f"Cannot compress vector of length {len(embedding)} to 1024")

    return embedding

def query_vector_index(index_name, query, searchtype, top_k_parameter):
    vector = generate_embeddings(query, compress_to_1024=False)
    search_client = SearchClient(AI_SEARCH_ENDPOINT, index_name, AzureKeyCredential(AI_SEARCH_KEY))
    vector_query = VectorizedQuery(vector=vector, fields="contentVector")
    # searchtypeがvector_onlyの場合は、search_textをNoneにする
    if searchtype == "Vector_only":
        search_text = None
    # searchtypeがvector_only以外の場合は、search_textにqueryを設定する
    else:
        search_text = query

    # searchtypeがvector_onlyもしくはHybridの場合
    if searchtype == "Vector_only" or searchtype == "Hybrid":
        results = search_client.search(search_text=search_text, vector_queries=[vector_query], top=int(top_k_parameter))
    # searchtypeがFullの場合
    else:
        results = search_client.search(search_text=search_text, vector_queries=[vector_query], top=int(top_k_parameter),
                                       query_type='semantic', semantic_configuration_name=AI_SEACH_SEMANTIC)

    return results

# chat履歴を Cosmos DB に保存する
def add_to_cosmos(item):
    container.upsert_item(item)

def add_feedback_to_cosmos(item):
    container_feedback.upsert_item(item)

def randomname(n):
    randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
    return ''.join(randlst)

def query_image_index(query, top_k=3):
    vector = generate_embeddings(query, compress_to_1024=True)  # ← 圧縮ON！

    image_search_endpoint = os.getenv("AI_SEARCH_IMAGE_ENDPOINT")
    image_search_key = os.getenv("AI_SEARCH_IMAGE_KEY")
    image_index_name = os.getenv("AI_SEARCH_IMAGE_INDEX_NAME")

    if not all([image_search_endpoint, image_search_key, image_index_name]):
        raise ValueError("画像用AI Searchの環境変数が正しく設定されていません。")

    search_client = SearchClient(
        endpoint=image_search_endpoint,
        index_name=image_index_name,
        credential=AzureKeyCredential(image_search_key)
    )

    vector_query = VectorizedQuery(vector=vector, fields="content_embedding")  # ← 画像用のフィールド
    try:
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top_k
        )
        return [r["content_path"] for r in results]
    except HttpResponseError as e:
        print("HTTP Response Error!")
        print(f"Status Code: {e.status_code}")
        print(f"Message: {e.message}")
        print(f"Details: {e.response.text}")
        raise
    

def get_image_from_image_blob(image_path: str) -> str:
    """
    BLOBストレージ上の画像ファイルを一時ファイルに保存し、そのパスを返す。
    image_pathは、相対パス（コンテナ内）か、完全なURLどちらも対応。
    """
    if not image_path:
        raise ValueError("image_path is empty. Check AI Search index content.")

    # フルURLの場合
    if image_path.startswith("http"):
        blob_client = BlobClient.from_blob_url(blob_url=image_path)
    else:
        # 相対パスの場合：事前に取得した image_blob_container_client を使う
        blob_client = image_blob_container_client.get_blob_client(blob=image_path)

    # 一時ファイルへ保存
    suffix = os.path.splitext(image_path)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        blob_stream = blob_client.download_blob()
        blob_stream.readinto(tmp)
        return tmp.name


def upload_file_to_blob_storage(uploaded_file):
    blob_client = blob_container_client.get_blob_client(uploaded_file.name)
    blob_client.upload_blob(uploaded_file.getbuffer(), overwrite=True)
    print(f"Uploaded {uploaded_file.name} to Blob Storage.")

def download_file_from_blob_storage(file_name):
    blob_client = blob_container_client.get_blob_client(file_name)
    download_file_path = os.path.join(tempfile.gettempdir(), file_name)
    with open(download_file_path, "wb") as download_file:
        download_data = blob_client.download_blob()
        download_data.readinto(download_file)
    print(f"Downloaded {file_name} from Blob Storage to {download_file_path}.")
    return download_file_path


#インデックス済みファイル名取得
def get_indexed_filenames(index_name):
    endpoint = os.getenv("AI_SEARCH_ENDPOINT")  # 例: https://xxx.search.windows.net
    api_key = os.getenv("AI_SEARCH_KEY")
    api_version = "2023-10-01-Preview"

    url = f"{endpoint}/indexes/{index_name}/docs/search?api-version={api_version}"

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    payload = {
        "search": "*",
        "select": "fileName",  # または "metadata_storage_name"
        "top": 1000
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    docs = response.json().get("value", [])
    return [doc.get("fileName", "NoName") for doc in docs]

def display_filenames_in_sidebar(filenames):
    st.sidebar.markdown("### 📄 回答可能なファイル")

    st.sidebar.markdown("""
        <style>
        .sidebar-scroll {
            max-height: 200px;
            overflow-y: auto;
            padding: 0.5rem;
            font-size: 0.85rem;
            background-color: var(--secondary-background-color);
            border: 1px solid var(--secondary-background-color);
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    html = "<div class='sidebar-scroll'>"
    for i, name in enumerate(filenames, 1):
        html += f"{i}. 📄 {name}<br>"
    html += "</div>"

    st.sidebar.markdown(html, unsafe_allow_html=True)

def main():
    if "feedback_status" not in st.session_state or not isinstance(st.session_state["feedback_status"], dict):
        st.session_state["feedback_status"] = {} 
    if "last_qid" not in st.session_state:
        st.session_state["last_qid"] = None
     # 固定値を直接変数に代入（UIは表示のみ）
    indexname = os.getenv("AI_SEARCH_INDEX_NAME")
    search_type = "Semantic_Hybrid"
    top_k_parameter = str(top_k_temp)
    Temperature_temp = 0.0
    SystemRole = SystemPrompt

    # Set page title and icon
    st.set_page_config(page_title="Kuraray RAG アプリ", page_icon="💬", layout="wide")

    # Display title
    st.markdown("# Kuraray G-SAPマニュアル Q&Aアプリ")

    # Display explanation in sidebar
    st.sidebar.header("Kuraray G-SAPマニュアル Q&Aアプリ")
    st.sidebar.markdown("OneNoteに記載されたマニュアルを検索できます。")
    
    centered_html = """
        <style>
        .center-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 70vh;
            text-align: center;
        }
        .center-container p {
            font-size: 20px;
            margin-bottom: 20px;
        }
        .center-container a {
            font-size: 18px;
            color: #1f77b4;
            text-decoration: none;
            margin: 5px 0;
        }
        .center-container a:hover {
            text-decoration: underline;
        }
        </style>

        <div class="center-container">
            <p>詳細は下記のリンク先から確認してください。以下のリンク先の情報を参考に回答しています。</p>
            <a href="https://kurarayglobal.sharepoint.com/:x:/s/krspp3/Eez5lJ1HHgBBgxy1h1s3K2IBz5gxagHng3kBNjeuK8qmSw" target="_blank">▶ QA集リスト</a>
            <a href="https://kurarayglobal.sharepoint.com/sites/krspp3/Business%20manual/Wave3%20JAPAN/07_Cross/QA%E9%9B%86!!%E7%B7%A8%E9%9B%86%E4%B8%8D%E5%8F%AF!!?d=w68ebcaa560ec4df396b00af9eae23c6d" target="_blank">▶ QA集</a>
            <a href="https://kurarayglobal.sharepoint.com/:f:/s/krspp3/EkJspieiifVGvz60d9oIJlYBHzpPKwSe8HZz_RO3mIri7A" target="_blank">▶ 業務マニュアル</a>
        </div>
        """

    # HTMLを埋め込む
    st.markdown(centered_html, unsafe_allow_html=True)
     # --- インデックス内のファイル名を表示（スクロール対応） ---
    #try:
        #filenames = get_indexed_filenames(indexname)
        #display_filenames_in_sidebar(filenames)
    #except Exception as e:
        #st.sidebar.error("ファイル名の取得に失敗しました。")
        #st.sidebar.exception(e)

    # セッションIDの初期化
    if "session_id" not in st.session_state:
        st.session_state['session_id'] = randomname(10)

    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state['messages'] = []

    # ファイル処理済みフラグの初期化
    if 'file_processed' not in st.session_state:
        st.session_state['file_processed'] = False
    
    for _ in range(5):  # 数字を変えることで下げ具合を調整
        st.sidebar.write("")

    # クリアボタンを押した場合、チャットとst.text_input,promptallをクリアする。
    if st.sidebar.button("Clear Chat"):
        st.session_state['messages'] = []
        promptall = ""
        # 新しいセッションIDを生成して保存
        st.session_state['session_id'] = randomname(10)
        # ファイル処理済みフラグをリセット
        st.session_state['file_processed'] = False
        # アプリを再実行してウィジェットをリセット
        st.rerun()

    # チャット履歴の表示
    messages = st.session_state.get('messages', [])
    for message in messages:
        # roleがassistantだったら、assistantのchat_messageを使う
        if message['role'] == 'assistant':
            with st.chat_message('assistant'):
                st.markdown(message['content'])
        # roleがuserだったら、userのchat_messageを使う
        elif message['role'] == 'user':
            with st.chat_message('user'):
                st.markdown(message['content'])
        else:  # 何も出力しない
            pass

    # Add system role to session state
    if SystemRole:
        # 既にroleがsystemのメッセージがある場合は、追加しない。ない場合は追加する。
        if not any(message["role"] == "system" for message in st.session_state.messages):
            st.session_state.messages.append({"role": "system", "content": SystemRole})

    # Azure AI Search のクライアントを作成する
    credential = AzureKeyCredential(AI_SEARCH_KEY)
    index_client = SearchIndexClient(
        endpoint=AI_SEARCH_ENDPOINT,
        credential=credential,
        api_version=AI_SEARCH_API_VERSION,
    )

    # ユーザからの入力を取得する
    if user_input := st.chat_input("プロンプトを入力してください"):
        # 検索する。search_fieldsはcontentを対象に検索する
        results = query_vector_index(indexname, user_input, search_type, top_k_parameter)

        # 変数を初期化する
        prompt_source = ""
        sourcetemp = []

        with st.chat_message("user"):
            st.markdown(user_input)

        # st.session_state.messagesの内容を平文にして、conversion_historyに代入する。RoleがSystemの場合は、代入しない。
        # 各messageのcontentを改行して表示する。roleもわかるように代入する
        conversion_history = ""
        for message in st.session_state.messages:
            if message['role'] == 'system':
                pass
            else:
                conversion_history += message['role'] + ": " + message['content'] + "\n\n"

        # resultsから各resultの結果を変数prompt_sourceに代入する。filepathとcontentの情報を代入する。
        for result in results:
            Score = result['@search.score']
            filename = result['fileName'] + "-" + str(result['chunkNo'])
            chunkNo = result['chunkNo']
            content = result['content']
            title = result['title']
            Keywords = result['keywords']

            # 変数prompt_sourceに各変数の値を追加する
            prompt_source += f"## filename: {filename}\n\n  ### score: {Score}\n\n  ### content: \n\n {content}\n\n"

            # filename, title, contentの内容をmarkdown形式でsourcetemp配列に格納する
            # sourcetempはresultの内容が変わる度に配列を変更する
            sourcetemp.append(f"## filename: {filename}\n\n  ### title: {title}\n\n  ### content: \n\n {content}\n\n")

        # プロンプトを作成する
        promptall = SystemRole + "\n\n# Sources(情報源): \n\n" + prompt_source + "# 今までの会話履歴：\n\n" + conversion_history + "# 回答の生成\n\nそれでは、制約を踏まえて最高の回答をしてください。あなたならできる！"
        st.session_state.messages.append({"role": "user", "content": user_input})

        # expanderを作成する
        #with st.sidebar.expander("プロンプトの表示"):
            # マークダウンを表示する
            #st.markdown(promptall)
        st.session_state["last_user_input"] = user_input
        
        st.session_state["last_prompt_source"] = prompt_source

        #Json形式のmessagestemp変数にroleをuserとして、promptallを代入する
        messagestemp = []
        messagestemp.append({"role": "system", "content": promptall})
        messagestemp.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            output = client.chat.completions.create(
                model=AZURE_OPENAI_CHAT_MODEL,
                messages=messagestemp,
                temperature=Temperature_temp,
                max_tokens=AZURE_OPENAI_CHAT_MAX_TOKENS,
                stream=True,
            )
            response = st.write_stream(output)
        
        qid = randomname(16)
        st.session_state["last_qid"] = qid
        st.session_state["feedback_status"][qid] = False
        

        st.session_state["last_response"] = response
        st.session_state.messages.append({"role": "assistant", "content": response})
            #output内に[]形式がある場合は、[]内のファイル名を取得し、sourcetemp内のfilenameと一致するものを探索する
            #一致するものがあれば、sourcetemp内の内容を表示する。既に1回表示されている場合は、2回目以降は表示しない
        with st.expander("参照元"):
            displayed_files = []  # 既に表示されたファイル名を追跡するためのリスト
            if "[" in response:
                filename = re.findall(r'\[(.*?)\]', response)
                for i in range(len(filename)):
                    for j in range(len(sourcetemp)):
                        if filename[i] in sourcetemp[j] and filename[i] not in displayed_files:  # ファイル名が既に表示されていないことを確認
                            with st.popover(filename[i]):
                                st.write(sourcetemp[j])
                            displayed_files.append(filename[i])  # ファイル名を追跡リストに追加
            else:
                pass
        related_images = query_image_index(user_input)

        if related_images:
            st.markdown("### 📷 関連画像:")
            cols = st.columns(len(related_images))

            for idx, image_path in enumerate(related_images):
                local_image_path = get_image_from_image_blob(image_path)
                cols[idx].image(local_image_path, caption=os.path.basename(image_path))
        else:
            st.markdown("関連する画像は見つかりませんでした。")

        # idにはランダム値を挿入する
        id1 = randomname(20)
        id2 = randomname(20)
        id3 = randomname(20)
        id4 = randomname(20)
        

        # チャット履歴を Cosmos DB に保存する。
        add_to_cosmos({"id": id1, "session": st.session_state['session_id'], "role": "user", "content": user_input})
        add_to_cosmos({"id": id2, "session": st.session_state['session_id'], "role": "assistant", "content": response})
        add_to_cosmos({"id": id3, "session": st.session_state['session_id'], "role": "context", "content": prompt_source}) 
        add_to_cosmos({"id": id4, "session": st.session_state['session_id'], "role": "eval", "question": user_input, "answer": response, "context": prompt_source, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}) 
    
    qid = st.session_state.get("last_qid")
    feedback_status = st.session_state.get("feedback_status", {})
    if st.session_state.get("last_response") and not feedback_status.get(st.session_state["last_qid"]):
        st.markdown("#### この回答は参考になりましたか？")
        id5 = randomname(20)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Good", key=f"good_{st.session_state['session_id']}"):
                add_feedback_to_cosmos({"id": id5, "session": st.session_state['session_id'], "role": "feedback", "feedback-type": "good",  "question":  st.session_state["last_user_input"], "answer": st.session_state["last_response"], "context": st.session_state["last_prompt_source"], "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                st.success("フィードバックありがとうございました！")
                st.session_state["feedback_status"][qid] = True 
                st.rerun()

        with col2:
            if st.button("👎 Bad", key=f"bad_{st.session_state['session_id']}"):
                add_feedback_to_cosmos({"id": id5, "session": st.session_state['session_id'], "role": "feedback", "feedback-type": "bad", "question":  st.session_state["last_user_input"], "answer": st.session_state["last_response"], "context": st.session_state["last_prompt_source"],  "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")} )                                     
                st.success("フィードバックありがとうございました！")
                st.session_state["feedback_status"][qid] = True
                st.rerun()
                
        
if __name__ == '__main__':
    main()
