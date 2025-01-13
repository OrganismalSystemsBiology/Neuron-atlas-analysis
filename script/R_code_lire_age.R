print("Start R code")

library(MASS)

# コマンドライン引数の取得
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]  # 入力された.binファイルのパス
output_file <- args[2]  # 出力ファイルのパス
num_data_points_per_group <- as.numeric(args[3])  # 1グループあたりのデータポイント数

print(input_file)

# ファイルの読み込み準備
con <- file(input_file, "rb")
file_size <- file.info(input_file)$size
itemsize <- 4 * 8  # 4 fields, each double size is 8 bytes

# データの読み込み
data_list <- list()
while (TRUE) {
    data <- readBin(con, what = double(), n = num_data_points_per_group * 4, endian = "little")
    if (length(data) == 0) break
    # データを行列に変換し、次にデータフレームに変換
    matrix_data <- matrix(data, ncol = 4, byrow = TRUE)
    colnames(matrix_data) <- c("Grid_id", "Disease_factor", "Age", "NeuN_neighbor_r_100um_diff")
    data_list[[length(data_list) + 1]] <- as.data.frame(matrix_data)
}
close(con)

# すべてのデータフレームを1つに結合
all_data <- do.call(rbind, data_list)

results <- lapply(split(all_data, all_data$Grid_id), function(group_data) {
    result <- tryCatch({
        fit <- lm(NeuN_neighbor_r_100um_diff ~ Age, data = group_data)
        coef <- coef(fit)["Age"]
        p_value <- summary(fit)$coefficients["Age", "Pr(>|t|)"]
        list(coef = coef, p_value = p_value)
    }, error = function(e) {
        print(paste("Error fitting model for group ID", group_data$Grid_id[1], ":", e$message))
        list(coef = 0.0, p_value = 1.0)
    })
    return(result)
})

# 結果の保存
save_results_to_bin <- function(results, file_path) {
    results_vector <- unlist(lapply(results, function(x) c(x$coef, x$p_value)))
    con <- file(file_path, "wb")
    writeBin(as.double(results_vector), con, endian = "little")
    close(con)
}

# 結果の保存
save_results_to_bin(results, output_file)

# 終了メッセージを出力
print(paste("Write End:", output_file))
