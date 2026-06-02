import numpy as np

def thuat_toan_bland(A, b, c):
    """
    Giải bài toán QHTT dạng chuẩn (Min, Ax <= b, x >= 0, b >= 0) bằng thuật toán Đơn hình.
    Áp dụng Quy tắc Bland để chống lặp (Cycling).
    """
    m, n = A.shape  # m: số ràng buộc, n: số biến
    
    # 1. LẬP BẢNG ĐƠN HÌNH BAN ĐẦU
    # Bảng có kích thước (m + 1) x (n + m + 1)
    # Gồm: 1 dòng Z + m dòng ràng buộc
    # Cột: n biến gốc + m biến bù (slack) + 1 cột vế phải (RHS)
    tableau = np.zeros((m + 1, n + m + 1))
    
    # Dòng Z (Row 0): Vì ta giải bài toán Min Z = cx, khởi tạo bằng c. 
    # (Nếu c_j < 0 thì ta chọn đưa vào cơ sở để làm giảm Z)
    tableau[0, :n] = c
    
    # Các dòng ràng buộc
    tableau[1:, :n] = A
    tableau[1:, n:n+m] = np.eye(m)  # Ma trận đơn vị cho các biến bù
    tableau[1:, -1] = b             # Cột vế phải
    
    # Lưu vết các biến cơ sở hiện tại (ban đầu là các biến bù s1..sm)
    # Chỉ số biến gốc: 0 đến n-1. Chỉ số biến bù: n đến n+m-1
    basic_vars = list(range(n, n + m))

    # 2. VÒNG LẶP ĐƠN HÌNH
    while True:
        # --- BƯỚC A: CHỌN BIẾN VÀO (QUY TẮC BLAND) ---
        # Tìm các hệ số < 0 trên dòng Z (sai số -1e-9 để tránh lỗi dấu phẩy động)
        c_bar = tableau[0, :-1]
        entering_cands = np.where(c_bar < -1e-9)[0]
        
        if len(entering_cands) == 0:
            break  # Không còn hệ số âm -> Đã đạt phương án tối ưu!
            
        # QUY TẮC BLAND 1: Chọn biến VÀO có chỉ số nhỏ nhất
        entering_col = entering_cands[0] 
        
        # --- BƯỚC B: CHỌN BIẾN RA (QUY TẮC BLAND) ---
        col_vals = tableau[1:, entering_col]
        rhs_vals = tableau[1:, -1]
        
        ratios = []
        valid_rows = []
        
        for i in range(m):
            if col_vals[i] > 1e-9:  # Chỉ xét các phần tử > 0
                ratios.append(rhs_vals[i] / col_vals[i])
                valid_rows.append(i)
                
        if not valid_rows:
            # Nếu tất cả phần tử cột VÀO đều <= 0, bài toán không giới nội
            return None, float('-inf')
            
        min_ratio = min(ratios)
        
        # Lọc ra các dòng cùng đạt min_ratio (để xử lý trường hợp suy biến)
        leaving_cands_rows = [valid_rows[k] for k, r in enumerate(ratios) if abs(r - min_ratio) < 1e-9]
        
        # QUY TẮC BLAND 2: Nếu có nhiều biến đạt min_ratio, chọn biến cơ sở có CHỈ SỐ NHỎ NHẤT đi ra
        leaving_row_idx = min(leaving_cands_rows, key=lambda r_idx: basic_vars[r_idx])
        leaving_row_tableau = leaving_row_idx + 1  # Bù 1 vì tableau có thêm dòng Z ở index 0
        
        # --- BƯỚC C: XOAY MA TRẬN (KHỬ GAUSS-JORDAN) ---
        basic_vars[leaving_row_idx] = entering_col  # Cập nhật danh sách biến cơ sở
        
        pivot_val = tableau[leaving_row_tableau, entering_col]
        tableau[leaving_row_tableau, :] /= pivot_val  # Chia dòng xoay cho phần tử trục
        
        # Biến các phần tử khác trên cột xoay thành 0
        for i in range(m + 1):
            if i != leaving_row_tableau:
                tableau[i, :] -= tableau[i, entering_col] * tableau[leaving_row_tableau, :]

    # 3. TRÍCH XUẤT KẾT QUẢ
    # Tạo mảng nghiệm cho n biến gốc (không lấy m biến bù)
    nghiem_chuan = np.zeros(n) 
    for i, var_idx in enumerate(basic_vars):
        if var_idx < n:  # Chỉ lấy giá trị nếu nó là biến gốc
            nghiem_chuan[var_idx] = tableau[i + 1, -1]
            
    # Giá trị tối ưu là trừ của góc dưới cùng bên phải dòng Z
    f_optimal = -tableau[0, -1]
    
    return nghiem_chuan, f_optimal