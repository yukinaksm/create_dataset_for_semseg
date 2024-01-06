import os # オペレーティングシステムへのアクセス
import random # ランダム機能へのアクセス
import shutil # 高度なファイル操作へのアクセス
from concurrent.futures import ThreadPoolExecutor # マルチスレッド機能へのアクセス

## すべてのサブディレクトリのパスを取得する関数
def get_subdirectories(directory):
    return [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

## サブディレクトリをシャッフルして、指定された比率で分割し、目的のディレクトリにコピーするメイン関数
def shuffle_and_copy_directories(source_directory, target_directory):
    # サブディレクトリのパスの取得
    subdirs = get_subdirectories(source_directory)
    #　パスの順序をシャッフル
    random.shuffle(subdirs)
    # 分割ポイントを計算(train:val:test=3:1:1)
    num_train = int(len(subdirs) * 0.6)
    num_val = int(len(subdirs) * 0.2)
    # ディレクトリの分割
    train_dirs = subdirs[:num_train]
    val_dirs = subdirs[num_train:num_train + num_val]
    test_dirs = subdirs[num_train + num_val:]
    # マルチスレッドでコピー
    with ThreadPoolExecutor(max_workers=4) as executor:
        for dir_type, dirs in zip(['train', 'val', 'test'], [train_dirs, val_dirs, test_dirs]):
            # コピー先のディレクトリパス
            dest = os.path.join(target_directory, dir_type)   
            # 必要に応じてディレクトリを作成
            os.makedirs(dest, exist_ok=True)
            # コピーの実行
            for subdir in dirs:
                executor.submit(shutil.copytree, subdir, os.path.join(dest, os.path.basename(subdir)))


# コピー元とコピー先のディレクトリの指定
source_directory = r"$HIP\PointNet2_custom_dataset\pdg_dataset"
target_directory = r"$HIP\PointNet2_custom_dataset\S3DIS_custom_dataset"

# メイン関数を実行
shuffle_and_copy_directories(source_directory, target_directory)
