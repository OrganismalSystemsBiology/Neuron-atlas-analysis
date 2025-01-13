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

# Processing each group by Grid_id
results <- lapply(split(all_data, all_data$Grid_id), function(group_data) {
    # Split by Age and sort by Age
    sorted_ages <- sort(unique(group_data$Age))
    age_groups <- split(group_data, group_data$Age)
    
    sapply(sorted_ages, function(age) {
        age_data <- age_groups[[as.character(age)]]
        valid_data <- subset(age_data, NeuN_neighbor_r_100um > 0)
        
        if (is.null(valid_data) || nrow(valid_data) == 0) {
            return(0.0)
        } else {
            return(mean(valid_data$NeuN_neighbor_r_100um))
        }
    })
})

# Flattening the results to save
flattened_results <- unlist(results)

# 結果の保存
save_results_to_bin <- function(results, file_path) {
    con <- file(file_path, "wb")
    writeBin(as.double(results), con, endian = "little")
    close(con)
}

# 結果の保存
save_results_to_bin(flattened_results, output_file)

# 終了メッセージを出力
print(paste("Write End:", output_file))
