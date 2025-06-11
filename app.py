import os
import gradio as gr
from PIL import Image
import numpy as np
import tensorflow as tf
from openai import OpenAI
from io import BytesIO
import base64
from datetime import datetime
from huggingface_hub import hf_hub_download

MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    print("模型不存在，開始下載…")
    import gdown
    url = "https://drive.google.com/uc?export=download&id=1UMEwZPIRXZufay438TXjsHbkftiXHFLQ"
    gdown.download(url, MODEL_PATH, quiet=False)
else:
    print("本地已存在模型，跳過下載。")

# －－－ 讀取環境變數 －－－
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")  # 如果你用 Groq 的話
os.environ['OPENAI_API_KEY'] = GROQ_API_KEY

# －－－ 載入模型與初始化 OpenAI 客戶端 －－－
model_cnn = tf.keras.models.load_model("model.h5")
client = OpenAI(base_url="https://api.groq.com/openai/v1")

# －－－ 類別名稱對照表 －－－
english_names = [
    "African Violet (Saintpaulia ionantha)", "Aloe Vera", "Anthurium (Anthurium andraeanum)",
    "Areca Palm (Dypsis lutescens)", "Asparagus Fern (Asparagus setaceus)", "Begonia (Begonia spp.)",
    "Bird of Paradise (Strelitzia reginae)", "Birds Nest Fern (Asplenium nidus)",
    "Boston Fern (Nephrolepis exaltata)", "Calathea", "Cast Iron Plant (Aspidistra elatior)",
    "Chinese Money Plant (Pilea peperomioides)", "Chinese evergreen (Aglaonema)",
    "Christmas Cactus (Schlumbergera bridgesii)", "Chrysanthemum", "Ctenanthe",
    "Daffodils (Narcissus spp.)", "Dracaena", "Dumb Cane (Dieffenbachia spp.)",
    "Elephant Ear (Alocasia spp.)", "English Ivy (Hedera helix)", "Hyacinth (Hyacinthus orientalis)",
    "Iron Cross begonia (Begonia masoniana)", "Jade plant (Crassula ovata)", "Kalanchoe",
    "Lilium (Hemerocallis)", "Lily of the valley (Convallaria majalis)",
    "Money Tree (Pachira aquatica)", "Monstera Deliciosa (Monstera deliciosa)", "Orchid",
    "Parlor Palm (Chamaedorea elegans)", "Peace lily", "Poinsettia (Euphorbia pulcherrima)",
    "Polka Dot Plant (Hypoestes phyllostachya)", "Ponytail Palm (Beaucarnea recurvata)",
    "Pothos (Ivy arum)", "Prayer Plant (Maranta leuconeura)", "Rattlesnake Plant (Calathea lancifolia)",
    "Rubber Plant (Ficus elastica)", "Sago Palm (Cycas revoluta)", "Schefflera",
    "Snake plant (Sanseviera)", "Tradescantia", "Tulip", "Venus Flytrap", "Yucca",
    "ZZ Plant (Zamioculcas zamiifolia)"
]
chinese_labels = [
    '非洲堇(非洲紫羅蘭)', '蘆薈', '火鶴花(紅掌、花燭)',
    '散尾葵(黃椰子)', '文竹(雲片竹、山草、雞絨芝)','秋海棠',
    '鶴望蘭(天堂鳥、極樂鳥花)', '鳥巢蕨(巢蕨、台灣山蘇、山蘇花、台灣山蘇花、鳥巢芒)',
    '波士頓腎蕨(波士頓蕨)', '疊苞竹芋','蜘蛛抱蛋(一葉蘭、粽葉、山豬耳、葉蘭)',
    '鏡面草', '粗肋草(廣東萬年青)',
    '聖誕仙人掌', '菊花', '密花竹芋',
    '水仙', '龍血樹(虎斑木)', '黛粉葉(萬年青、花葉萬年青、啞蕉、啞巴甘蔗)',
    '海芋(姑婆芋)', '常春藤(洋常春藤、長春藤、土鼓藤、木蔦、百角蜈蚣)', '風信子',
    '鐵甲秋海棠(鐵十字秋海棠、毛葉秋海棠、刺毛秋海棠、馬蹄海棠)','翡翠木(發財樹、玉樹、燕子掌)', '長壽花(家樂花、矮生伽藍菜、聖誕伽藍菜、壽星花)',
    '百合(萱草)', '鈴蘭(山谷百合、風鈴草、君影草)',
    '馬拉巴栗(發財樹、招財樹、錢樹、美國花生、光瓜栗、瓜栗)', '龜背芋(龜背竹、鳳梨蕉、蓬萊蕉、電信蘭)', '蘭花',
    '袖珍椰子(客廳棕櫚)', '白鶴芋(苞葉芋、和平百合)', '聖誕紅(一品紅、聖誕花)',
    '嫣紅蔓(粉點木、紅點草)', '酒瓶蘭(馬尾棕櫚)',
    '黃金葛(黃金藤、萬年青、綠蘿)','豹紋竹芋(祈禱花、豹斑竹芋、綠脈竹芋)', '響尾蛇竹芋(箭羽竹芋)',
    '橡膠榕(印度榕、印度尼西亞橡膠榕、印度橡膠榕、印度橡膠樹、印度橡樹)', '蘇鐵(琉球蘇鐵、臺東蘇鐵、鐵樹、鳳尾蕉、鳳尾松、避火樹)', '鵝掌藤(七葉蓮、七葉藤、狗腳蹄)',
    '虎尾蘭(虎皮蘭、錦蘭、黃尾蘭、岳母舌)','紫露草(吊竹梅)', '鬱金香', '捕蠅草', '絲蘭',
    '金錢樹(美鐵芋、金幣樹、雪鐵芋、澤米葉天南星、扎米蓮)'
]

# －－－ 輔助：PIL 圖片轉 Base64（用於 HTML <img>）－－－
def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

# －－－ 輔助：決定使用哪個植物名 －－－
def extract_or_use_last_plant(user_question: str, default_plant: str) -> str:
    system_prompt = f"""以下是使用者的問題：「{user_question}」
請你判斷這句話中是否有提到植物名稱。
若有，請直接回覆該植物名稱；若沒有，請只回覆「無」。"""
    messages = [
        {"role":"system","content":"你是一位植物小幫手，只回覆植物名稱或「無」。"},
        {"role":"user","content":system_prompt}
    ]
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    ).choices[0].message.content.strip()
    return default_plant if resp == "無" else resp

# －－－ 主流程：預測 + Chat 回答 －－－
last_plant = None

def predict_and_chat(image: Image.Image, question: str):
    global last_plant
    html = ""
    if image is not None:
        # 1. 圖像預處理 + CNN 預測
        img = image.convert("RGB")
        img_resized = img.resize((224,224))
        arr = np.expand_dims(np.array(img_resized)/255.0, axis=0)
        preds = model_cnn.predict(arr)[0]
        top5 = preds.argsort()[-5:][::-1]

        # 2. 建立 Top‑5 表格
        rows = ""
        for idx in top5:
            rows += f"<tr><td>{english_names[idx]}<br><b>{chinese_labels[idx]}</b></td><td>{preds[idx]:.2%}</td></tr>"
        top1_idx = top5[0]
        top1_chi = chinese_labels[top1_idx]
        top1_prob = preds[top1_idx]
        last_plant = top1_chi

        html += f"""
        <div style='display:flex;gap:20px;align-items:flex-start'>
          <div>
            <h3>🖼️ 圖片預覽</h3>
            <img src="{image_to_base64(img)}" width="200" style="border-radius:12px;"/>
          </div>
          <div>
            <h3>🌿 Top‑5 分類結果</h3>
            <table border="1" style="border-collapse:collapse;text-align:center">
              <tr><th>英文 / 中文</th><th>準確度</th></tr>
              {rows}
            </table>
          </div>
        </div>"""
        
        # 3. 生成植物介紹
        intro_prefix = ""
        if top1_prob < 0.6:
            intro_prefix = "雖然信心不高，但這張圖可能是：\n"
        prompt_intro = f"請用簡單介紹植物「{top1_chi}」，包含外觀、習性、常見用途、照顧方式（澆水頻率、光照需求）。"
        intro_resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"你是一位專業植物小幫手，使用繁體中文回答。"},
                {"role":"user","content":prompt_intro}
            ]
        ).choices[0].message.content

        html += f"""
        <div style='padding:10px;border:2px solid #90EE90;border-radius:12px;margin-top:10px'>
          <h3>📘 AI 植物簡介：{top1_chi}</h3>
          <div>{intro_prefix}{intro_resp.replace(chr(10),'<br>')}</div>
        </div>"""

        # 4. 若有追問，再做一次 Chat
        if question:
            plant_used = extract_or_use_last_plant(question, top1_chi)
            prompt_q = f"使用者問：「{question}」，植物是「{plant_used}」，請回答。"
            qa_resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":"你是一位專業植物小幫手，使用繁體中文回答。"},
                    {"role":"user","content":prompt_q}
                ]
            ).choices[0].message.content
            html += f"""
            <div style='padding:10px;border:2px solid #228B22;border-radius:12px;margin-top:10px'>
              <h3>🗨️ 你的問題：{question}</h3>
              <div>{qa_resp.replace(chr(10),'<br>')}</div>
            </div>"""

    else:
        html = "<p style='color:red;'>請先用相機拍照或上傳一張植物圖片。</p>"

    return html

# －－－ 建立 Gradio 介面 －－－
image_input = gr.Image(
    label="拍照或選圖",
    type="numpy"
)
text_input = gr.Textbox(
    label="詢問問題（可留空）",
    placeholder="例如：這盆蘆薈要怎麼澆水？"
)
output = gr.HTML()

with gr.Blocks() as demo:
    gr.Markdown("## 🌿 植物小幫手")
    with gr.Row():
        img = image_input
        txt = text_input
    btn = gr.Button("開始分析")
    btn.click(fn=predict_and_chat, inputs=[img, txt], outputs=output)
    demo.launch()
