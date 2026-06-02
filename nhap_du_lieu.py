import numpy as np

from hai_pha import thuat_toan_bland, thuat_toan_hai_pha
from hinh_hoc import phuong_phap_truot_ham_muc_tieu

class BaiToanQuyHoachTuyenTinh:

    def __init__(self):
        # Khởi tạo các thuộc tính cơ bản của bài toán
        self.so_bien_goc = 0  # Số biến ban đầu người dùng nhập
        self.he_so_f_goc = []  # Hệ số c của hàm mục tiêu ban đầu
        self.loai_muc_tieu = "min"  # 'max' hoặc 'min'

        # Danh sách lưu các ràng buộc chính ban đầu
        self.cac_rang_buoc = (
            []
        )  # Mỗi phần tử dạng: (mang_he_so_VT, dau_rang_buoc, gia_tri_VP)

        # Ràng buộc dấu ban đầu của từng biến: '>=0', '<=0', hoặc 'free'
        self.dau_cua_bien = []

        # Ma trận và vectơ sau khi đã đưa về DẠNG CHUẨN (Hàm MIN, ràng buộc <=, biến >= 0)
        self.A_chuan = None  # Ma trận hệ số ràng buộc dạng chuẩn
        self.b_chuan = None  # Vectơ vế phải dạng chuẩn
        self.c_chuan = None  # Hệ số hàm mục tiêu dạng chuẩn

        # Danh sách quản lý cách ánh xạ từ biến dạng chuẩn về biến gốc để xuất kết quả
        self.anh_xa_bien = []

    def buoc1_nhap_ham_muc_tieu(self):
        """Bước 1: Nhập hàm mục tiêu"""
        print("NHẬP HÀM MỤC TIÊU")
        self.loai_muc_tieu = (
            input("Bạn muốn tìm MIN hay MAX (nhập min/max): ")
            .strip()
            .lower()
        )
        while self.loai_muc_tieu not in ["max", "min"]:
            self.loai_muc_tieu = (
                input("Vui lòng chỉ nhập 'min' hoặc 'max': ").strip().lower()
            )

        self.so_bien_goc = int(input("Nhập số lượng biến của bài toán: "))

        print(
            "Nhập các hệ số của hàm mục tiêu (cách nhau bởi dấu cách, ví dụ: 3 -5 2):"
        )
        while True:
            try:
                self.he_so_f_goc = [
                    float(x) for x in input("-> c = ").split()
                ]
                if len(self.he_so_f_goc) == self.so_bien_goc:
                    break
                print(f"Số lượng hệ số phải bằng {self.so_bien_goc}!")
            except ValueError:
                print("Dữ liệu không hợp lệ. Vui lòng nhập lại số thực.")

    def buoc2_nhap_cac_rang_buoc(self):
        """Bước 2: Nhập các ràng buộc đẳng thức và bất đẳng thức"""
        print("\nNHẬP CÁC RÀNG BUỘC ĐẲNG THỨC/BẤT ĐẲNG THỨC")
        so_rang_buoc = int(input("Nhập số lượng ràng buộc đẳng thức/bất đẳng thức: "))

        print(
            "Với mỗi ràng buộc, nhập các hệ số vế trái, dấu (<=, >=, =), và vế phải."
        )
        print("Ví dụ: 1 2 -1 <= 10")

        for i in range(so_rang_buoc):
            while True:
                try:
                    dong_nhap = input(f"Ràng buộc {i+1}: ").split()
                    dau = dong_nhap[-2]
                    gia_tri_vp = float(dong_nhap[-1])
                    he_so_vt = [float(x) for x in dong_nhap[:-2]]

                    if len(he_so_vt) != self.so_bien_goc:
                        print(
                            f"Số hệ số vế trái ({len(he_so_vt)}) phải bằng số biến ({self.so_bien_goc})!"
                        )
                        continue
                    if dau not in ["<=", ">=", "="]:
                        print("Dấu phải là '<=', '>=', hoặc '='!")
                        continue

                    self.cac_rang_buoc.append((he_so_vt, dau, gia_tri_vp))
                    break
                except Exception:
                    print(
                        "Cú pháp sai! Vui lòng nhập đúng dạng (VD: 2 1.5 <= 4)"
                    )

    def buoc3_chuan_hoa_bai_toan(self):
        """Bước 3: Nhập ràng buộc dấu, đưa về dạng chuẩn (Min, <=, >=0)"""
        print("\nNHẬP RÀNG BUỘC DẤU")
        print("Nhập ràng buộc dấu cho từng biến ('>=0', '<=0', 'free'):")

        for i in range(self.so_bien_goc):
            while True:
                dau_bien = (
                    input(f"Biến x{i+1} là (>=0 / <=0 / free): ")
                    .strip()
                    .replace(" ", "")
                )
                if dau_bien in [">=0", "<=0", "free"]:
                    self.dau_cua_bien.append(dau_bien)
                    break
                print("Vui lòng nhập chính xác: '>=0', '<=0' hoặc 'free'")

        # --- ĐƯA BÀI TOÁN VỀ DẠNG CHUẨN ---
        he_so_f_sua = np.array(self.he_so_f_goc, dtype=float)
        if self.loai_muc_tieu == "max":
            he_so_f_sua = -he_so_f_sua

        A_khoi_tao = np.array(
            [dong_vt for dong_vt, _, _ in self.cac_rang_buoc], dtype=float
        )

        A_bien_doi = []
        c_bien_doi = []
        self.anh_xa_bien = []
        chi_so_bien_moi = 0

        for j in range(self.so_bien_goc):
            dau_x = self.dau_cua_bien[j]
            cot_A = A_khoi_tao[:, j]
            gia_tri_c = he_so_f_sua[j]

            if dau_x == "free":
                A_bien_doi.append(cot_A)
                c_bien_doi.append(gia_tri_c)
                idx_duong = chi_so_bien_moi
                chi_so_bien_moi += 1

                A_bien_doi.append(-cot_A)
                c_bien_doi.append(-gia_tri_c)
                idx_am = chi_so_bien_moi
                chi_so_bien_moi += 1

                self.anh_xa_bien.append(
                    {
                        "chi_so_goc": j,
                        "loai": "free",
                        "idx_duong": idx_duong,
                        "idx_am": idx_am,
                    }
                )
            elif dau_x == "<=0":
                A_bien_doi.append(-cot_A)
                c_bien_doi.append(-gia_tri_c)
                self.anh_xa_bien.append(
                    {
                        "chi_so_goc": j,
                        "loai": "negative",
                        "idx_moi": chi_so_bien_moi,
                    }
                )
                chi_so_bien_moi += 1
            elif dau_x == ">=0":
                A_bien_doi.append(cot_A)
                c_bien_doi.append(gia_tri_c)
                self.anh_xa_bien.append(
                    {
                        "chi_so_goc": j,
                        "loai": "normal",
                        "idx_moi": chi_so_bien_moi,
                    }
                )
                chi_so_bien_moi += 1

        A_bien_doi = np.column_stack(A_bien_doi)
        c_bien_doi = np.array(c_bien_doi, dtype=float)

        A_dong_cuoi = []
        b_dong_cuoi = []

        for i, (_, dau_rb, gia_tri_vp) in enumerate(self.cac_rang_buoc):
            dong_da_xu_ly_bien = A_bien_doi[i, :]

            if dau_rb == "<=":
                A_dong_cuoi.append(dong_da_xu_ly_bien)
                b_dong_cuoi.append(gia_tri_vp)
            elif dau_rb == ">=":
                A_dong_cuoi.append(-dong_da_xu_ly_bien)
                b_dong_cuoi.append(-gia_tri_vp)
            elif dau_rb == "=":
                A_dong_cuoi.append(dong_da_xu_ly_bien)
                b_dong_cuoi.append(gia_tri_vp)
                A_dong_cuoi.append(-dong_da_xu_ly_bien)
                b_dong_cuoi.append(-gia_tri_vp)

        self.A_chuan = np.array(A_dong_cuoi, dtype=float)
        self.b_chuan = np.array(b_dong_cuoi, dtype=float)
        self.c_chuan = c_bien_doi

    def hien_thi_dang_chuan(self):
        """ĐÃ SỬA: Đưa hàm ra ngoài, đứng ngang hàng cấu trúc lớp"""
        print("\n--- THÔNG TIN MA TRẬN DẠNG CHUẨN ---")
        print(f"Hàm mục tiêu cần MIN: Z = {self.c_chuan}")
        print("Ma trận ràng buộc chính A_chuan (tất cả đều mang dấu <=):")
        print(self.A_chuan)
        print(f"Véctơ vế phải b_chuan: {self.b_chuan}")

    def xac_dinh_phuong_phap_giai(self):
        co_b_am = np.any(self.b_chuan < 0)
        if co_b_am:
            print("Dùng phương pháp hai pha")
            return "hai_pha"
        else:
            print("Dùng phương pháp Bland")
            return "bland"

    def hoan_nguyen_nghiem_goc(self, nghiem_dang_chuan):
        nghiem_goc = np.zeros(self.so_bien_goc)
        for anh_xa in self.anh_xa_bien:
            idx_goc = anh_xa["chi_so_goc"]

            if anh_xa["loai"] == "normal":
                nghiem_goc[idx_goc] = nghiem_dang_chuan[anh_xa["idx_moi"]]
            elif anh_xa["loai"] == "negative":
                nghiem_goc[idx_goc] = -nghiem_dang_chuan[anh_xa["idx_moi"]]
            elif anh_xa["loai"] == "free":
                x_duong = nghiem_dang_chuan[anh_xa["idx_duong"]]
                x_am = nghiem_dang_chuan[anh_xa["idx_am"]]
                nghiem_goc[idx_goc] = x_duong - x_am
        return nghiem_goc

    def giai_va_xuat_ket_qua(self):
        phuong_phap = self.xac_dinh_phuong_phap_giai()

        # 1. Gọi thuật toán giải để lấy kết quả (Thêm biến trang_thai để nhận đủ 3 giá trị trả về)
        if phuong_phap == "bland":
            nghiem_chuan, f_optimal, trang_thai = thuat_toan_bland(
                self.A_chuan, self.b_chuan, self.c_chuan
            )
        else:
            
            nghiem_chuan, f_optimal, trang_thai = thuat_toan_hai_pha(
                self.A_chuan, self.b_chuan, self.c_chuan
            )

        # 2. RẼ NHÁNH ĐỂ IN KẾT QUẢ THEO TỪNG TRƯỜNG HỢP TOÁN HỌC

        # TRƯỜNG HỢP 1: VÔ NGHIỆM (Hàm giải trả về nghiem_chuan là None và f_optimal là None)
        if trang_thai == "vo_nghiem" or (nghiem_chuan is None and f_optimal is None):
            print("Bài toán vô nghiệm do miền chấp nhận được là rỗng.")
            if self.loai_muc_tieu == "min":
                print("    -> Giá trị tối ưu min = +inf (Dương vô cùng)")
            else:
                print("    -> Giá trị tối ưu max = -inf (Âm vô cùng)")

        # TRƯỜNG HỢP 2: KHÔNG BỊ CHẶN (Nghị lực hàm tiến ra vô cùng)
        elif trang_thai == "khong_gioi_noi" or (nghiem_chuan is None and (
            f_optimal == float("inf") or f_optimal == float("-inf"))
        ):
            print("Bài toán vô nghiệm do không giới nội")
            if self.loai_muc_tieu == "min":
                print("    -> Giá trị tối ưu min = -inf (Âm vô cùng)")
            else:
                print("    -> Giá trị tối ưu max = +inf (Dương vô cùng)")

        # TRƯỜNG HỢP 3: CÓ NGHIỆM TỐI ƯU DUY NHẤT (HOẶC ĐA NGHIỆM)
        else:
            # Tiến hành dịch nghiệm từ dạng chuẩn về biến gốc x ban đầu
            nghiem_goc_cuoi_cung = self.hoan_nguyen_nghiem_goc(nghiem_chuan)

            # Nếu đề bài ban đầu là MAX, ta đảo dấu lại giá trị f tối ưu vì lõi đang giải MIN
            if self.loai_muc_tieu == "max":
                f_optimal = -f_optimal

            print("Bài toán có nghiệm tối ưu")
            print(f"--> Giá trị tối ưu của hàm mục tiêu Z = {f_optimal}")
            print("--> Nghiệm tối ưu:")
            for idx, gia_tri in enumerate(nghiem_goc_cuoi_cung):
                print(f"    x{idx+1} = {gia_tri}")
    def phuong_phap_hinh_hoc(self):
        if self.so_bien_goc == 2:
            print(
                "\n--> Đây là bài toán hai biến nên có thể giải được bằng hình học - phương pháp trượt hàm mục tiêu."
            )
            print(
                "\n--> Bạn có muốn giải bài toán bằng phương pháp hình học này để đối chiếu không?"
            )
            lua_chon = (
                    input("Nhập lựa chọn của bạn (có/không): ")
                    .strip()
                    .lower()
            )
            if lua_chon == "có" :
                phuong_phap_truot_ham_muc_tieu(self)

if __name__ == "__main__":    
    bai_toan = BaiToanQuyHoachTuyenTinh()
    bai_toan.buoc1_nhap_ham_muc_tieu()
    bai_toan.buoc2_nhap_cac_rang_buoc()
    bai_toan.buoc3_chuan_hoa_bai_toan()
    bai_toan.hien_thi_dang_chuan()
    bai_toan.giai_va_xuat_ket_qua()
    bai_toan.phuong_phap_hinh_hoc()
