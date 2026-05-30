import numpy as np


import numpy as np

# =====================================================================
# HÀM 1: PHÉP XOAY MA TRẬN (KHỬ GAUSS-JORDAN)
# Mục đích: Đưa biến mới vào cơ sở bằng cách biến cột của nó thành 
# véc-tơ đơn vị (có số 1 tại phần tử trục, các vị trí khác bằng 0).
# =====================================================================
def xoay_ma_tran(bang, dong, cot):
    # Chia toàn bộ dòng trục cho phần tử trục để phần tử trục trở thành 1
    bang[dong] = bang[dong] / bang[dong, cot]
    
    # Duyệt qua các dòng còn lại trong ma trận
    for i in range(bang.shape[0]):
        if i != dong: # Bỏ qua dòng trục vừa xử lý
            # Lấy dòng hiện tại trừ đi một bội số của dòng trục 
            # để triệt tiêu phần tử cùng cột về 0
            bang[i] -= bang[i, cot] * bang[dong]


# =====================================================================
# HÀM 2: VÒNG LẶP ĐƠN HÌNH 
# Mục đích: Thực hiện quá trình lặp đi lặp lại việc chọn biến vào, 
# biến ra và xoay ma trận cho đến khi tìm được phương án tối ưu.
# =====================================================================
def chay_vong_lap_don_hinh(bang, co_so, dung_quy_tac_bland=False):
    so_cot = bang.shape[1] - 1  # Bỏ qua cột cuối cùng (vế phải b)
    so_dong = bang.shape[0] - 1 # Bỏ qua dòng cuối cùng (dòng đánh giá delta)
    
    while True:
        # --- BƯỚC 1: CHỌN BIẾN VÀO CƠ SỞ (Biến làm giảm hàm mục tiêu) ---
        if dung_quy_tac_bland:
            # Quy tắc Bland: Chọn biến đầu tiên từ trái sang có hệ số đánh giá âm
            # (Giúp chống lặp vòng vô hạn)
            cot_vao = -1
            for j in range(so_cot):
                if bang[-1, j] < -1e-9: # Dùng -1e-9 thay vì 0 để tránh sai số máy tính
                    cot_vao = j
                    break
            if cot_vao == -1:
                break # Không còn hệ số âm -> Đã tối ưu!
        else:
            # Quy tắc Dantzig tiêu chuẩn: Chọn biến có hệ số đánh giá âm NHẤT
            cot_vao = np.argmin(bang[-1, :-1])
            if bang[-1, cot_vao] >= -1e-9:
                break # Toàn bộ >= 0 -> Đã tối ưu!

        # --- BƯỚC 2: CHỌN BIẾN RA KHỎI CƠ SỞ (Điều kiện tỷ số nhỏ nhất) ---
        danh_sach_ty_so = []
        for i in range(so_dong):
            # Chỉ xét các phần tử > 0 trong cột xoay
            if bang[i, cot_vao] > 1e-9:
                # Tính tỷ số = vế phải / phần tử cột xoay
                danh_sach_ty_so.append(bang[i, -1] / bang[i, cot_vao])
            else:
                # Nếu <= 0, biến này có thể tăng vô hạn, gán tỷ số là vô cực
                danh_sach_ty_so.append(float('inf'))

        ty_so_nho_nhat = min(danh_sach_ty_so)
        
        # Nếu tất cả các tỷ số đều là vô cực, miền nghiệm không bị chặn
        if ty_so_nho_nhat == float('inf'):
            return "khong_gioi_noi" 

        # Xác định dòng ra (dòng chứa tỷ số nhỏ nhất)
        if dung_quy_tac_bland:
            # Nếu có nhiều tỷ số min bằng nhau, ưu tiên biến gốc có chỉ số nhỏ nhất
            ung_cu_vien = [i for i in range(so_dong) if abs(danh_sach_ty_so[i] - ty_so_nho_nhat) < 1e-9]
            dong_ra = min(ung_cu_vien, key=lambda i: co_so[i])
        else:
            dong_ra = np.argmin(danh_sach_ty_so)

        # --- BƯỚC 3: CẬP NHẬT MA TRẬN VÀ CƠ SỞ ---
        xoay_ma_tran(bang, dong_ra, cot_vao)
        co_so[dong_ra] = cot_vao # Ghi nhận biến mới vào cơ sở
        
    return "toi_uu"


# =====================================================================
# HÀM 3: THUẬT TOÁN HAI PHA (KỸ THUẬT 1 BIẾN GIẢ)
# Mục đích: Giải các bài toán có vế phải âm (không có sẵn cơ sở xuất phát)
# =====================================================================
def thuat_toan_hai_pha(ma_tran_A, ve_phai_b, he_so_c):
    so_phuong_trinh, so_bien = ma_tran_A.shape
    
    # --- THIẾT LẬP BẢNG PHA 1 ---
    # Kích thước: [Số dòng + 1 dòng đánh giá] x [Biến gốc + 1 Biến giả + Biến bù + Vế phải]
    bang = np.zeros((so_phuong_trinh + 1, so_bien + 1 + so_phuong_trinh + 1))
    
    # 1. Nạp ma trận A ban đầu
    bang[:so_phuong_trinh, :so_bien] = ma_tran_A
    # 2. Chèn 1 cột biến giả (hệ số luôn là -1 ở mọi phương trình)
    bang[:so_phuong_trinh, so_bien] = -1
    # 3. Chèn ma trận đơn vị cho các biến bù (slacks)
    bang[:so_phuong_trinh, so_bien + 1 : so_bien + 1 + so_phuong_trinh] = np.eye(so_phuong_trinh)
    # 4. Nạp cột vế phải
    bang[:so_phuong_trinh, -1] = ve_phai_b

    # Mặc định cơ sở ban đầu là các biến bù
    co_so = [so_bien + 1 + i for i in range(so_phuong_trinh)]

    # --- ÉP KHẢ THI TỨC THÌ ---
    # Tìm vế phải âm nhất
    chi_so_b_am_nhat = np.argmin(bang[:so_phuong_trinh, -1])
    if bang[chi_so_b_am_nhat, -1] < -1e-9:
        # Xoay đưa biến giả vào cơ sở thay cho biến bù ở phương trình đó.
        # Thao tác này lập tức đổi dấu toàn bộ các vế phải thành số dương.
        xoay_ma_tran(bang, chi_so_b_am_nhat, so_bien)
        co_so[chi_so_b_am_nhat] = so_bien

    # Cài đặt hàm mục tiêu Pha 1: Cực tiểu hóa biến giả (Hệ số = 1)
    bang[-1, so_bien] = 1
    
    # Chuẩn hóa dòng đánh giá (ép hệ số của các biến cơ sở về 0)
    for i, vi_tri_co_so in enumerate(co_so):
        bang[-1] -= bang[-1, vi_tri_co_so] * bang[i]

    # Giải Pha 1
    ket_qua_pha_1 = chay_vong_lap_don_hinh(bang, co_so)

    # --- KIỂM TRA KẾT QUẢ PHA 1 (TÌM XEM CÓ NGHIỆM KHÔNG) ---
    if so_bien in co_so: # Nếu biến giả vẫn nằm trong cơ sở
        dong_chua_x0 = co_so.index(so_bien)
        if bang[dong_chua_x0, -1] > 1e-9:
            # Giá trị biến giả > 0 -> Hệ ràng buộc mâu thuẫn -> Vô nghiệm
            return None, None, "vo_nghiem"
        else:
            # Trạng thái suy biến: Biến giả = 0 nhưng kẹt trong cơ sở
            # Tìm một phần tử khác 0 trên cùng dòng để làm trục xoay nhằm "đá" nó ra
            da_tim_thay_phan_tu_xoay = False
            for j in range(so_bien + 1 + so_phuong_trinh):
                if j != so_bien and abs(bang[dong_chua_x0, j]) > 1e-9:
                    xoay_ma_tran(bang, dong_chua_x0, j)
                    co_so[dong_chua_x0] = j
                    da_tim_thay_phan_tu_xoay = True
                    break
            # Nếu cả dòng toàn số 0 (0 = 0), đây là phương trình thừa, ta xóa nó đi
            if not da_tim_thay_phan_tu_xoay:
                bang = np.delete(bang, dong_chua_x0, axis=0)
                co_so.pop(dong_chua_x0)
                so_phuong_trinh -= 1

    # --- THIẾT LẬP VÀ GIẢI PHA 2 ---
    # Xóa hoàn toàn cột biến giả vì nó đã hết giá trị lợi dụng
    bang = np.delete(bang, so_bien, axis=1)
    # Cập nhật lại vị trí các cột trong mảng cơ sở do có 1 cột bị xóa
    co_so = [c if c < so_bien else c - 1 for c in co_so]

    # Khôi phục hàm mục tiêu gốc cần tối ưu
    bang[-1, :] = 0
    bang[-1, :so_bien] = he_so_c

    # Chuẩn hóa lại dòng đánh giá cho hàm mục tiêu gốc
    for i, vi_tri_co_so in enumerate(co_so):
        bang[-1] -= bang[-1, vi_tri_co_so] * bang[i]

    # Giải Pha 2
    ket_qua_pha_2 = chay_vong_lap_don_hinh(bang, co_so)
    
    if ket_qua_pha_2 == "khong_gioi_noi":
        return None, float('-inf'), "khong_gioi_noi"

    # --- RÚT TRÍCH KẾT QUẢ ---
    nghiem_toi_uu = np.zeros(so_bien)
    for i, vi_tri_co_so in enumerate(co_so):
        if vi_tri_co_so < so_bien: # Chỉ lấy nghiệm của các biến gốc (x)
            nghiem_toi_uu[vi_tri_co_so] = bang[i, -1]
    
    # Giá trị hàm mục tiêu luôn nằm ở góc dưới cùng bên phải, đảo dấu
    gia_tri_toi_uu = -bang[-1, -1]

    # --- KIỂM TRA ĐA NGHIỆM (VÔ SỐ NGHIỆM) ---
    trang_thai = "toi_uu_duy_nhat"
    so_cot_hien_tai = bang.shape[1] - 1
    for j in range(so_cot_hien_tai):
        # Nếu có biến nằm ngoài cơ sở mà hệ số đánh giá = 0, đưa nó vào 
        # sẽ tạo ra một nghiệm mới với cùng giá trị Z.
        if j not in co_so and abs(bang[-1, j]) < 1e-9:
            trang_thai = "vo_so_nghiem"
            break

    return nghiem_toi_uu, gia_tri_toi_uu, trang_thai


# =====================================================================
# HÀM 4: THUẬT TOÁN ĐƠN HÌNH (SỬ DỤNG LUẬT BLAND)
# Mục đích: Dùng cho bài toán đã có sẵn cơ sở (vế phải b >= 0 toàn bộ)
# =====================================================================
def thuat_toan_bland(ma_tran_A, ve_phai_b, he_so_c):
    so_phuong_trinh, so_bien = ma_tran_A.shape
    
    # Cấu trúc bảng tương tự Pha 2 nhưng làm ngay từ đầu
    bang = np.zeros((so_phuong_trinh + 1, so_bien + so_phuong_trinh + 1))
    bang[:so_phuong_trinh, :so_bien] = ma_tran_A
    bang[:so_phuong_trinh, so_bien : so_bien + so_phuong_trinh] = np.eye(so_phuong_trinh)
    bang[:so_phuong_trinh, -1] = ve_phai_b
    bang[-1, :so_bien] = he_so_c

    co_so = [so_bien + i for i in range(so_phuong_trinh)]
    
    # Chuẩn hóa dòng đánh giá
    for i, vi_tri_co_so in enumerate(co_so):
        bang[-1] -= bang[-1, vi_tri_co_so] * bang[i]

    # Giải thuật toán (Thuật toán Bland để chống xoay vòng)
    ket_qua = chay_vong_lap_don_hinh(bang, co_so, dung_quy_tac_bland=True)
    
    if ket_qua == "khong_gioi_noi":
        return None, float('-inf'), "khong_gioi_noi"

    # Rút trích nghiệm
    nghiem_toi_uu = np.zeros(so_bien)
    for i, vi_tri_co_so in enumerate(co_so):
        if vi_tri_co_so < so_bien:
            nghiem_toi_uu[vi_tri_co_so] = bang[i, -1]
    
    gia_tri_toi_uu = -bang[-1, -1]
    
    # Kiểm tra vô số nghiệm
    trang_thai = "toi_uu_duy_nhat"
    so_cot_hien_tai = bang.shape[1] - 1
    for j in range(so_cot_hien_tai):
        if j not in co_so and abs(bang[-1, j]) < 1e-9:
            trang_thai = "vo_so_nghiem"
            break
            
    return nghiem_toi_uu, gia_tri_toi_uu, trang_thai
