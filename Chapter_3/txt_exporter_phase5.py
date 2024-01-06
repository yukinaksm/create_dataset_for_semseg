#### PHASE 5 ####



import hou # houdiniへのアクセス
import os #オペレーティングシステムへのアクセス

geo_cache = {} # 点群をキャッシュするための辞書の作成

### 点群をエクスポートするためのメイン関数
def export_point_cloud_to_txt(whole_node_path, extracted_node_path, param_node_path, work_item):
    ## 内部関数1：処理に用いる点群をキャッシュする
    def get_or_cache_geometry(node_path):
        if node_path not in geo_cache:
            node = hou.node(node_path)
            if node is None:
                raise ValueError(f"Node not found: {node_path}")
            # パスをキーとして点群を辞書に保存する
            geo_cache[node_path] = node.geometry()
        return geo_cache[node_path]

    ## 内部関数2：sopノードのパラメータの更新
    def update_sop_parameters(node_path):
        # 指定ノードへのアクセス
        extracted_node = hou.node(node_path)
        if extracted_node is None:
            raise ValueError(f"Node not found: {node_path}")

        # 各work_itemからアトリビュートを取得し、extractedパラメータに値を反映する
        for attrib_name in ['B3_posy', 'A2_posy', 'A1_posy', 'B4_HIJIKI_num', 'B3_HIJIKI_num', 'A2_HIJIKI_num', 'A1_HIJIKI_num', 'class_extract', 'class_index', 'valx']:
            if work_item.hasAttrib(attrib_name):
                attrib_value = work_item.attribValue(attrib_name)
                if extracted_node.parm(attrib_name) is not None:
                    extracted_node.parm(attrib_name).set(attrib_value)

    ## 内部関数3：txt形式でのエクスポート設定
    def save_geometry_data(geo, file_path):
        # 既存のファイルは上書きしない
        if os.path.exists(file_path):
            return
        # キャッシュされた点群の色情報を取得
        color_attrib = geo.findPointAttrib("Cd")
        # 指定されたファイルパスにディレクトリを作成（既存なら無視）
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # txtファイルの書き込み
        with open(file_path, 'w') as file:
            # 各点に対してループ処理で情報を格納する
            for point in geo.points():
                # 位置情報
                pos = point.position()
                # 色情報の初期化
                color = (0, 0, 0)
                # 0-255のレンジでRGB値を設定
                if color_attrib:
                    color = point.attribValue("Cd")
                    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                # 書き込み. 位置情報は小数点以下第3位までで記録
                file.write(f"{pos.x():.3f} {pos.y():.3f} {pos.z():.3f} {color[0]} {color[1]} {color[2]}\n")
        # ファイル容量が0の場合、ファイルを削除
        if os.path.exists(file_path) and os.path.getsize(file_path) == 0:
            os.remove(file_path)

    ## 内部関数終わり


    # extractedノードのパラメータを更新
    update_sop_parameters(param_node_path)

    # 各アトリビュートの定義
    B3_posy = work_item.attribValue("B3_posy") 
    A2_posy = work_item.attribValue("A2_posy") 
    A1_posy = work_item.attribValue("A1_posy") 
    B4_HIJIKI_num = work_item.attribValue("B4_HIJIKI_num") 
    B3_HIJIKI_num = work_item.attribValue("B3_HIJIKI_num") 
    A2_HIJIKI_num = work_item.attribValue("A2_HIJIKI_num") 
    A1_HIJIKI_num = work_item.attribValue("A1_HIJIKI_num") 
    class_extract = work_item.attribValue("class_extract") 
    valx = work_item.attribValue("valx") 
    class_index = work_item.attribValue("class_index")

    # ファイルパスの生成
    base_path = hou.expandString("$HIP/PointNet2_custom_dataset/pdg_dataset/")
    common_path = f'{B3_posy}_{A2_posy}_{A1_posy}pos__{A1_HIJIKI_num}_{A2_HIJIKI_num}_{B4_HIJIKI_num}_{B3_HIJIKI_num}num__{valx}val'
    whole_file_path = f'{base_path}{common_path}/{common_path}.txt'
    extracted_file_path = f'{base_path}{common_path}/Annotations/{class_extract}_{class_index}.txt'

    # 点群を保存
    whole_geo = get_or_cache_geometry(whole_node_path)
    save_geometry_data(whole_geo, whole_file_path)
    extracted_geo = get_or_cache_geometry(extracted_node_path)
    save_geometry_data(extracted_geo, extracted_file_path)

# PDGコンテキスト内での実行の例
whole_node_path = '/obj/jodoji_like_kumimono/Whole'  # 全体点群のパス
extracted_node_path = '/obj/jodoji_like_kumimono/Extracted'   # 部材点群のパス
param_node_path = '/obj/jodoji_like_kumimono/PARAMS1'   # パラメータコントローラーのパス

# メイン関数を実行
export_point_cloud_to_txt(whole_node_path, extracted_node_path, param_node_path, work_item)
