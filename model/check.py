def find_break_positions(df):
    # 데이터프레임을 datetime 인덱스를 기준으로 정렬
    df_sorted = df.sort_index()

    # 인덱스 간격 계산 (분 단위로 변환)
    index_diff = df_sorted.index.to_series().diff().dt.seconds // 60

    # 끊김이 발생한 위치 찾기
    break_positions = []
    for i in range(1, len(index_diff)):
        if index_diff.iloc[i] > 1:
            break_positions.append(i)

    if break_positions:
        print(f"끊김이 발생한 위치들: {break_positions}")
        print(f"끊긴 시간의 총 군데수: {len(break_positions)}")  # 끊긴 시간의 총 군데수 출력
        for position in break_positions:
            if position > 0 and position < len(df_sorted.index) - 1:
                previous_index = df_sorted.index[position - 1]
                current_index = df_sorted.index[position]
                next_index = df_sorted.index[position + 1]
                print(f"끊김이 발생한 위치 {position}:")
                print(f"이전 인덱스: {previous_index}")
                print(f"현재 인덱스: {current_index}")
                print(f"다음 인덱스: {next_index}")
                print()
            else:
                print(f"끊김이 발생한 위치 {position}에 대한 이전 또는 다음 인덱스가 존재하지 않습니다.")
    else:
        print("끊김이 발생한 위치가 없습니다.")

def handle_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index

    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
    print("Number of Outliers", len(outliers))

    return outliers