import sys
import numpy as np
import copy
import open3d as o3d

#パス1
# path="../3rdparty/Open3D/examples/test_data/ICP/"
# source= o3d.io.read_point_cloud (path+"cloud_bin_0.pcd")
# target= o3d.io.read_point_cloud (path+"cloud_bin_1.pcd")

#パス2
path="/Users/nakashimayuki/Library/CloudStorage/GoogleDrive-zhongdaoy189@gmail.com/マイドライブ/jodoji/o3d/"
source= o3d.io.read_point_cloud (path+"masu.pcd")
target= o3d.io.read_point_cloud (path+"kumimono.pcd")

#初期化
source.paint_uniform_color([0.5,0.5,1])
target.paint_uniform_color([1,0.5,0.5])
initial_trans= np.identity (4)
initial_trans [0,3] = -3.0


#関数
def draw_registration_result(source, target, transformation):
	pcds= list()
	for s in source:
		temp = copy.deepcopy(s)
		pcds.append(temp.transform (transformation))
	pcds += target
	o3d.visualization.draw_geometries (pcds, zoom=0.3199,
		front = [0.024,0.225,0.973],
		lookat = [0.488, 1.722, 1.556],
		up = [0.047, -0.972, 0.226]
	)


def curvature_based_keypoint_extraction(pcd, radius, curvature_threshold):
    # 点群の法線推定
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # KDTreeの構築
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # 曲率の計算
    curvature = np.zeros(len(pcd.points))
    for i, point in enumerate(pcd.points):
        # KDTreeを使用して最近傍点を探索
        _, knn_indices, _ = pcd_tree.search_knn_vector_3d(point, 30)
        knn_points = np.asarray(pcd.points)[knn_indices]
        covariance = np.cov(knn_points.T)
        eigenvalues = np.linalg.eigvals(covariance)
        curvature[i] = np.min(eigenvalues) / np.sum(eigenvalues)

    # 曲率が閾値以上の点を特徴点として選択
    keypoints_indices = np.where(curvature > curvature_threshold)[0]
    keypoints = o3d.geometry.PointCloud()
    keypoints.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[keypoints_indices])

    return keypoints


def create_lineset_from_correspondences( corrs_set, pcd1, pcd2, 
                                         transformation=np.identity(4) ):
    """ 対応点セットからo3d.geometry.LineSetを作成する．
    Args:
        result(o3d.utility.Vector2iVector) ): 対応点のidセット
        pcd1(o3d.geometry.PointCloud): resultを計算した時の点群1
        pcd2(o3d.geometry.PointCloud): resultを計算した時の点群2
        transformation(numpy.ndarray): 姿勢変換行列(4x4)
    Return:
        o3d.geometry.LineSet
    """
    pcd1_temp = copy.deepcopy(pcd1)
    pcd1_temp.transform(transformation) 
    corrs = np.asarray(corrs_set)
    np_points1 = np.array(pcd1_temp.points)
    np_points2 = np.array(pcd2.points)
    points = list()
    lines = list()

    for i in range(corrs.shape[0]):
        points.append( np_points1[corrs[i,0]] )
        points.append( np_points2[corrs[i,1]] )
        lines.append([2*i, (2*i)+1])

    colors = [np.random.rand(3) for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def compute_fpfh_feature(pcd, radius):
    # 点群の法線を推定する
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # FPFH特徴量を計算
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )
    return fpfh

# 新しい関数: 物理的に重複しない対応点を探す
def find_non_overlapping_correspondences(s_kp, t_kp, np_s_feature, np_t_feature, threshold, max_correspondences):
    non_overlapping_corrs = []
    used_indices = set()

    for _ in range(max_correspondences):
        corrs = o3d.utility.Vector2iVector()
        for i, feat in enumerate(np_s_feature):
            if i in used_indices:
                continue
            distance = np.linalg.norm(np_t_feature - feat, axis=1)
            nearest_idx = np.argmin(distance)
            dist_order = np.argsort(distance)
            ratio = distance[dist_order[0]] / distance[dist_order[1]]
            if ratio < threshold and nearest_idx not in used_indices:
                corrs.append(np.array([[i], [nearest_idx]], np.int32))
                used_indices.add(i)
                used_indices.add(nearest_idx)

        if len(corrs) == 0:
            break

        non_overlapping_corrs.append(corrs)

    return non_overlapping_corrs



######################################################################################

# 描画
# draw_registration_result([source], [target], initial_trans)

# 曲率ベースの特徴点抽出
radius = 0.01  # 法線計算のための半径
curvature_threshold = 0.02  # 曲率の閾値
s_kp= curvature_based_keypoint_extraction(source, radius, curvature_threshold)
t_kp= curvature_based_keypoint_extraction(target, radius, curvature_threshold)

s_kp.paint_uniform_color([0,1,0])
t_kp.paint_uniform_color([0,1,0])
# draw_registration_result([source,s_kp], [target,t_kp], initial_trans)

# FPFH特徴量の計算
radius_feature = 0.05  # 特徴量計算のための半径
s_feature = compute_fpfh_feature(s_kp, radius_feature)
t_feature = compute_fpfh_feature(t_kp, radius_feature)

# 特徴量データをNumPy配列に変換
np_s_feature = np.asarray(s_feature.data).T
np_t_feature = np.asarray(t_feature.data).T

# 対応点探索
corrs = o3d.utility.Vector2iVector()
threshold = 0.9
for i,feat in enumerate(np_s_feature):
	# source側の特定の特徴量とtarget側の全特徴量間のノルムを計算
	distance = np.linalg.norm( np_t_feature - feat, axis=1 )
	nearest_idx = np.argmin(distance)
	dist_order = np.argsort(distance)
	ratio = distance[dist_order[0]] / distance[dist_order[1]]
	if ratio < threshold:
		corr = np.array( [[i],[nearest_idx]], np.int32 )
		corrs.append( corr )

print('対応点セットの数：', (len(corrs)) )



# 変更点: RANSACの実行と結果の評価
exclusion_radius = 0.1  # 近接除外パラメータ
num_masu = 4
max_correspondences = num_masu  # 物理的に重複しない対応点の最大数
distance_threshold = 0.02
found_correspondences = find_non_overlapping_correspondences(s_kp, t_kp, np_s_feature, np_t_feature, threshold, max_correspondences)

# 異なる色を割り当てるためのカラーリスト
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]  # 赤、緑、青、黄色

# 近接除外パラメータ
proximity_threshold = 0.15  # この値は実際のスケールに合わせて調整してください

# 変換後のソース点群が近すぎるかを判断する関数
def is_too_close(source_list, new_source, threshold):
    for src in source_list:
        dist = np.linalg.norm(np.asarray(src.points).mean(axis=0) - np.asarray(new_source.points).mean(axis=0))
        if dist < threshold:
            return True
    return False

# RANSACの結果を保存するためのリスト
transformed_sources = []
def exclude_close_points_to_transformed_source(transformed_source, target_kp, exclusion_radius):
    # KDTreeを構築
    target_kp_tree = o3d.geometry.KDTreeFlann(target_kp)

    transformed_source_center = np.asarray(transformed_source.points).mean(axis=0)
    [k, idx, _] = target_kp_tree.search_radius_vector_3d(transformed_source_center, exclusion_radius)

    # 除外する点以外の点を取得
    to_keep = set(range(len(target_kp.points))) - set(idx)
    filtered_points = [target_kp.points[i] for i in to_keep]
    return o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(filtered_points))

# RANSACと結果の評価
while len(transformed_sources) < num_masu:
    found_correspondences = find_non_overlapping_correspondences(
        s_kp, t_kp, np_s_feature, np_t_feature, threshold, max_correspondences
    )

    for corrs in found_correspondences:
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            s_kp, t_kp, corrs, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], 
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )

        # 変換後のソース点群を作成
        source_temp = copy.deepcopy(source)
        source_temp.transform(result.transformation)
        source_temp.paint_uniform_color(colors[len(transformed_sources) % len(colors)])
        transformed_sources.append(source_temp)

        # 結果の評価（変換行列、対応点の数、など）
        print(f"Match {len(transformed_sources)} - Transformation matrix:\n{result.transformation}")
        print(f"Match {len(transformed_sources)} - Number of correspondences: {len(result.correspondence_set)}")

        # t_kpから近接点を除外
        t_kp = exclude_close_points_to_transformed_source(source_temp, t_kp, proximity_threshold)

        if len(transformed_sources) >= num_masu:
            break

# 最終的な描画：全ての変換後のソース点群とターゲット点群を表示
if transformed_sources:
    o3d.visualization.draw_geometries(transformed_sources + [target], window_name="All Transformed Sources and Target")
else:
    print("No transformed sources to display.")
