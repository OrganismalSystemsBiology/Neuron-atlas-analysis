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
itemsize <- 4 * 4  # 4 fields, each integer size is 4 bytes

# データの読み込み
data_list <- list()
while (TRUE) {
    data <- readBin(con, what = integer(), n = num_data_points_per_group * 4, endian = "little")
    if (length(data) == 0) break
    # Convert the data to a matrix, then to a data frame
    matrix_data <- matrix(data, ncol = 4, byrow = TRUE)
    colnames(matrix_data) <- c("Grid_id", "Disease_factor", "Age", "NeuN_neighbor_r_100um")
    data_list[[length(data_list) + 1]] <- as.data.frame(matrix_data)
}
close(con)

# Combining all data frames into one
all_data <- do.call(rbind, data_list)

results <- lapply(split(all_data, all_data$Grid_id), function(group_data) {
    valid_data <- subset(group_data, NeuN_neighbor_r_100um > 0)
    
    if (sum(valid_data$NeuN_neighbor_r_100um) <= 0 || nrow(valid_data) < 6) {
        return(list(coef = 0.0, p_value = 1.0))
    } else {
        result <- tryCatch({
            fit <- glm.nb(NeuN_neighbor_r_100um ~ Age, data = valid_data)
            coef <- coef(fit)["Age"]
            p_value <- summary(fit)$coefficients["Age", "Pr(>|z|)"]
            list(coef = coef, p_value = p_value)
        }, error = function(e) {
            print(paste("Error fitting model for group ID", group_data$Grid_id[1], ":", e$message))
            list(coef = 0.0, p_value = 1.0)
        }, finally = {
            # ここには通常、リソースのクリーンアップや、必ず実行したいログ記録などが含まれます
            #print(paste("Completed processing for group ID", group_data$Grid_id[1]))
        })
        return(result)
    }
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
