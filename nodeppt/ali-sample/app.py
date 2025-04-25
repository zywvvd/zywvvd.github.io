from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from lib import SimpleTexOcr
import tempfile
import os

app = Flask(__name__)

# 配置文件保存路径
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
simpletex_ocr_obj = SimpleTexOcr()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'image' not in request.files:
        return jsonify({'error': '未接收到图片'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '空文件名'}), 400

    try:
        # 关键修改1：使用 tempfile 创建临时文件
        with tempfile.NamedTemporaryFile(
            suffix='.jpg',      # 指定扩展名
            delete=False,       # 关闭后不自动删除（以便后续处理）
            dir=os.getenv('TEMP', '.')) as tmp_file:  # 使用系统临时目录
            
            # 关键修改2：直接写入二进制流（无需保存到固定路径）
            file_bytes = file.stream.read()
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name  # 获取系统生成的唯一路径

        # 关键修改3：直接传递文件路径给 OCR 处理
        # （假设 SimpleTexOcr.process 方法接收文件路径）
        ocr_result = simpletex_ocr_obj.query(tmp_path)

        # 关键修改4：处理完成后主动删除临时文件
        os.unlink(tmp_path)

        return jsonify({
            'result': ocr_result
        })

    except Exception as e:
        # 异常时确保删除临时文件
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
