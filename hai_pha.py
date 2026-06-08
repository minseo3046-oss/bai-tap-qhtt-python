import numpy as np

def trich_xuat_dinh(bang, co_so, so_bien_goc=2):
    dinh = [0.0] * so_bien_goc
    for i, vi_tri_co_so in enumerate(co_so):
        if vi_tri_co_so < so_bien_goc:
            dinh[vi_tri_co_so] = bang[i, -1]
    return tuple(np.round(dinh, 6))

def kiem_tra_xoay_vong(lich_su_co_so, co_so_hien_tai):
    trang_thai = tuple(sorted(co_so_hien_tai))
    if trang_thai in lich_su_co_so:
        return True
    lich_su_co_so.append(trang_thai)
    return False

def xoay_ma_tran(bang, dong, cot):
    bang[dong] = bang[dong] / bang[dong, cot]
    for i in range(bang.shape[0]):
        if i != dong:
            bang[i] -= bang[i, cot] * bang[dong]

# ---> ĐÃ THÊM THAM SỐ lich_su_bang <---
def chay_vong_lap_don_hinh(bang, co_so, dung_quy_tac_bland=False, lich_su_dinh=None, so_bien_goc=2, lich_su_co_so=None, lich_su_bang=None):
    so_cot = bang.shape[1] - 1
    so_dong = bang.shape[0] - 1
    so_buoc = 0
    
    while True:
        # === CHỤP ẢNH MA TRẬN TRƯỚC KHI XOAY ===
        if lich_su_bang is not None:
            lich_su_bang.append({
                "bang": np.copy(bang),
                "co_so": list(co_so)
            })

        if lich_su_dinh is not None:
            dinh_hien_tai = trich_xuat_dinh(bang, co_so, so_bien_goc)
            if not lich_su_dinh or lich_su_dinh[-1] != dinh_hien_tai:
                lich_su_dinh.append(dinh_hien_tai)
                
        if dung_quy_tac_bland:
            cot_vao = -1
            for j in range(so_cot):
                if bang[-1, j] < -1e-9:
                    cot_vao = j
                    break
            if cot_vao == -1: break 
        else:
            cot_vao = np.argmin(bang[-1, :-1])
            if bang[-1, cot_vao] >= -1e-9: break

        danh_sach_ty_so = []
        for i in range(so_dong):
            if bang[i, cot_vao] > 1e-9:
                danh_sach_ty_so.append(bang[i, -1] / bang[i, cot_vao])
            else:
                danh_sach_ty_so.append(float('inf'))

        ty_so_nho_nhat = min(danh_sach_ty_so)
        if ty_so_nho_nhat == float('inf'): return "khong_gioi_noi"

        if dung_quy_tac_bland:
            ung_cu_vien = [i for i in range(so_dong) if abs(danh_sach_ty_so[i] - ty_so_nho_nhat) < 1e-9]
            dong_ra = min(ung_cu_vien, key=lambda i: co_so[i])
        else:
            dong_ra = np.argmin(danh_sach_ty_so)

        xoay_ma_tran(bang, dong_ra, cot_vao)
        co_so[dong_ra] = cot_vao
        so_buoc += 1
        
        if lich_su_co_so is not None:
            if kiem_tra_xoay_vong(lich_su_co_so, co_so): return "xoay_vong"

    # === CHỤP ẢNH MA TRẬN TỐI ƯU CUỐI CÙNG ===
    if lich_su_bang is not None:
        lich_su_bang.append({
            "bang": np.copy(bang),
            "co_so": list(co_so)
        })

    if lich_su_dinh is not None:
        dinh_cuoi = trich_xuat_dinh(bang, co_so, so_bien_goc)
        if not lich_su_dinh or lich_su_dinh[-1] != dinh_cuoi:
            lich_su_dinh.append(dinh_cuoi)

    return "toi_uu", so_buoc
        
def thuat_toan_hai_pha(ma_tran_A, ve_phai_b, he_so_c):
    so_phuong_trinh, so_bien = ma_tran_A.shape
    duong_di_toa_do = [] if so_bien == 2 else None
    lich_su_bang = [] # ---> Khởi tạo danh sách lưu bảng
    
    bang = np.zeros((so_phuong_trinh + 1, so_bien + 1 + so_phuong_trinh + 1))
    bang[:so_phuong_trinh, :so_bien] = ma_tran_A
    bang[:so_phuong_trinh, so_bien] = -1
    bang[:so_phuong_trinh, so_bien + 1 : so_bien + 1 + so_phuong_trinh] = np.eye(so_phuong_trinh)
    bang[:so_phuong_trinh, -1] = ve_phai_b
    co_so = [so_bien + 1 + i for i in range(so_phuong_trinh)]

    chi_so_b_am_nhat = np.argmin(bang[:so_phuong_trinh, -1])
    if bang[chi_so_b_am_nhat, -1] < -1e-9:
        if duong_di_toa_do is not None:
            duong_di_toa_do.append(trich_xuat_dinh(bang, co_so, so_bien))
        xoay_ma_tran(bang, chi_so_b_am_nhat, so_bien)
        co_so[chi_so_b_am_nhat] = so_bien

    bang[-1,:] = 0
    bang[-1] -= bang[chi_so_b_am_nhat]
    bang[-1,so_bien] = 0
    
    for i, vi_tri_co_so in enumerate(co_so):
        bang[-1] -= bang[-1, vi_tri_co_so] * bang[i]
    
    lich_su_co_so=[]
    ket_qua_pha_1, so_buoc1 = chay_vong_lap_don_hinh(bang, co_so, dung_quy_tac_bland=False, lich_su_dinh=duong_di_toa_do, so_bien_goc=so_bien, lich_su_co_so=lich_su_co_so, lich_su_bang=lich_su_bang)

    if so_bien in co_so:
        dong_chua_x0 = co_so.index(so_bien)
        if bang[dong_chua_x0, -1] > 1e-9:
            duong_di_toa_do = []
            return None, None, "vo_nghiem", duong_di_toa_do, so_buoc1, lich_su_bang # Trả về thêm lich_su_bang
        else:
            da_tim_thay_phan_tu_xoay = False
            for j in range(so_bien + 1 + so_phuong_trinh):
                if j != so_bien and abs(bang[dong_chua_x0, j]) > 1e-9:
                    xoay_ma_tran(bang, dong_chua_x0, j)
                    co_so[dong_chua_x0] = j
                    da_tim_thay_phan_tu_xoay = True
                    break
            if not da_tim_thay_phan_tu_xoay:
                bang = np.delete(bang, dong_chua_x0, axis=0)
                co_so.pop(dong_chua_x0)
                so_phuong_trinh -= 1

    bang = np.delete(bang, so_bien, axis=1)
    co_so = [c if c < so_bien else c - 1 for c in co_so]

    bang[-1, :] = 0
    bang[-1, :so_bien] = he_so_c
    for i, vi_tri_co_so in enumerate(co_so):
        bang[-1] -= bang[-1, vi_tri_co_so] * bang[i]

    ket_qua_pha_2, so_buoc_2 = chay_vong_lap_don_hinh(bang, co_so, dung_quy_tac_bland=False, lich_su_dinh=duong_di_toa_do, so_bien_goc=so_bien, lich_su_co_so=lich_su_co_so, lich_su_bang=lich_su_bang)
    
    if ket_qua_pha_2 == "khong_gioi_noi":
        duong_di_toa_do=[]
        return None, float('-inf'), "khong_gioi_noi", duong_di_toa_do, so_buoc_2, lich_su_bang

    nghiem_A = np.zeros(so_bien)
    for i, vi_tri_co_so in enumerate(co_so):
        if vi_tri_co_so < so_bien:
            nghiem_A[vi_tri_co_so] = bang[i, -1]
            
    gia_tri_toi_uu = -bang[-1, -1]
    dinh_toi_uu = [tuple(np.round(nghiem_A, 6))] 
    trang_thai = "toi_uu_duy_nhat"

    so_cot_hien_tai = bang.shape[1] - 1
    for j in range(so_cot_hien_tai):
        if j not in co_so and abs(bang[-1, j]) < 1e-9:
            danh_sach_ty_so = []
            for i in range(bang.shape[0] - 1):
                if bang[i, j] > 1e-9:
                    danh_sach_ty_so.append(bang[i, -1] / bang[i, j])
                else:
                    danh_sach_ty_so.append(float('inf'))
            ty_so_min = min(danh_sach_ty_so)
            if ty_so_min != float('inf'):
                dinh_B = np.zeros(so_bien)
                for i in range(len(co_so)):
                    var_idx = co_so[i]
                    if var_idx < so_bien:
                        dinh_B[var_idx] = bang[i, -1] - ty_so_min * bang[i, j]
                dinh_tuple = tuple(np.round(dinh_B, 6))
                if dinh_tuple not in dinh_toi_uu:
                    dinh_toi_uu.append(dinh_tuple)
                    trang_thai = "vo_so_nghiem"

    nghiem_tra_ve = [np.array(d) for d in dinh_toi_uu]
    # Trả về 5 biến
    return nghiem_tra_ve, gia_tri_toi_uu, trang_thai, duong_di_toa_do, 0, lich_su_bang

def thuat_toan_bland(ma_tran_A, ve_phai_b, he_so_c):
    so_phuong_trinh, so_bien = ma_tran_A.shape
    duong_di_toa_do = [] if so_bien == 2 else None
    lich_su_bang = []
    
    bang = np.zeros((so_phuong_trinh + 1, so_bien + so_phuong_trinh + 1))
    bang[:so_phuong_trinh, :so_bien] = ma_tran_A
    bang[:so_phuong_trinh, so_bien : so_bien + so_phuong_trinh] = np.eye(so_phuong_trinh)
    bang[:so_phuong_trinh, -1] = ve_phai_b
    bang[-1, :so_bien] = he_so_c
    co_so = [so_bien + i for i in range(so_phuong_trinh)]
    
    for i, vi_tri_co_so in enumerate(co_so):
        bang[-1] -= bang[-1, vi_tri_co_so] * bang[i]
    
    lich_su_co_so=[]
    trang_thai, so_buoc_lap = chay_vong_lap_don_hinh(bang, co_so, dung_quy_tac_bland=True, lich_su_dinh=duong_di_toa_do, so_bien_goc=so_bien, lich_su_co_so=lich_su_co_so, lich_su_bang=lich_su_bang)
    
    if trang_thai == "khong_gioi_noi":
        duong_di_toa_do=[]
        return None, float('-inf'), "khong_gioi_noi", duong_di_toa_do, so_buoc_lap, lich_su_bang

    nghiem_A = np.zeros(so_bien)
    for i, vi_tri_co_so in enumerate(co_so):
        if vi_tri_co_so < so_bien:
            nghiem_A[vi_tri_co_so] = bang[i, -1]
            
    gia_tri_toi_uu = -bang[-1, -1]
    dinh_toi_uu = [tuple(np.round(nghiem_A, 6))] 
    trang_thai = "toi_uu_duy_nhat"

    so_cot_hien_tai = bang.shape[1] - 1
    for j in range(so_cot_hien_tai):
        if j not in co_so and abs(bang[-1, j]) < 1e-9:
            danh_sach_ty_so = []
            for i in range(bang.shape[0] - 1):
                if bang[i, j] > 1e-9:
                    danh_sach_ty_so.append(bang[i, -1] / bang[i, j])
                else:
                    danh_sach_ty_so.append(float('inf'))
            ty_so_min = min(danh_sach_ty_so)
            if ty_so_min != float('inf'):
                dinh_B = np.zeros(so_bien)
                for i in range(len(co_so)):
                    var_idx = co_so[i]
                    if var_idx < so_bien:
                        dinh_B[var_idx] = bang[i, -1] - ty_so_min * bang[i, j]
                dinh_tuple = tuple(np.round(dinh_B, 6))
                if dinh_tuple not in dinh_toi_uu:
                    dinh_toi_uu.append(dinh_tuple)
                    trang_thai = "vo_so_nghiem"

    nghiem_tra_ve = [np.array(d) for d in dinh_toi_uu]
    return nghiem_tra_ve, gia_tri_toi_uu, trang_thai, duong_di_toa_do, so_buoc_lap, lich_su_bang
    
def thuat_toan_dantzig(ma_tran_A, ve_phai_b, he_so_c):
    so_phuong_trinh, so_bien = ma_tran_A.shape
    duong_di_toa_do = [] if so_bien == 2 else None
    lich_su_bang = []
    
    bang = np.zeros((so_phuong_trinh + 1, so_bien + so_phuong_trinh + 1))
    bang[:so_phuong_trinh, :so_bien] = ma_tran_A
    bang[:so_phuong_trinh, so_bien : so_bien + so_phuong_trinh] = np.eye(so_phuong_trinh)
    bang[:so_phuong_trinh, -1] = ve_phai_b
    bang[-1, :so_bien] = he_so_c
    co_so = [so_bien + i for i in range(so_phuong_trinh)]
    
    for i, vi_tri_co_so in enumerate(co_so):
        bang[-1] -= bang[-1, vi_tri_co_so] * bang[i]
    lich_su_co_so = []    

    trang_thai, so_buoc_lap = chay_vong_lap_don_hinh(bang, co_so, dung_quy_tac_bland=False, lich_su_dinh=duong_di_toa_do, so_bien_goc=so_bien, lich_su_co_so=lich_su_co_so, lich_su_bang=lich_su_bang)
    
    if trang_thai == "khong_gioi_noi":
        duong_di_toa_do = []
        return None, float('-inf'), "khong_gioi_noi", duong_di_toa_do, so_buoc_lap, lich_su_bang
    if trang_thai == "xoay_vong":
        return None, None, "xoay_vong", [],so_buoc_lap, lich_su_bang

    nghiem_A = np.zeros(so_bien)
    for i, vi_tri_co_so in enumerate(co_so):
        if vi_tri_co_so < so_bien:
            nghiem_A[vi_tri_co_so] = bang[i, -1]
            
    gia_tri_toi_uu = -bang[-1, -1]
    dinh_toi_uu = [tuple(np.round(nghiem_A, 6))] 
    trang_thai = "toi_uu_duy_nhat"

    so_cot_hien_tai = bang.shape[1] - 1
    for j in range(so_cot_hien_tai):
        if j not in co_so and abs(bang[-1, j]) < 1e-9:
            danh_sach_ty_so = []
            for i in range(bang.shape[0] - 1):
                if bang[i, j] > 1e-9:
                    danh_sach_ty_so.append(bang[i, -1] / bang[i, j])
                else:
                    danh_sach_ty_so.append(float('inf'))
            ty_so_min = min(danh_sach_ty_so)
            if ty_so_min != float('inf'):
                dinh_B = np.zeros(so_bien)
                for i in range(len(co_so)):
                    var_idx = co_so[i]
                    if var_idx < so_bien:
                        dinh_B[var_idx] = bang[i, -1] - ty_so_min * bang[i, j]
                dinh_tuple = tuple(np.round(dinh_B, 6))
                if dinh_tuple not in dinh_toi_uu:
                    dinh_toi_uu.append(dinh_tuple)
                    trang_thai = "vo_so_nghiem"

    nghiem_tra_ve = [np.array(d) for d in dinh_toi_uu]
    return nghiem_tra_ve, gia_tri_toi_uu, trang_thai, duong_di_toa_do,so_buoc_lap, lich_su_bang