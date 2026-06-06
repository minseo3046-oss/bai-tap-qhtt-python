import matplotlib.pyplot as plt
import numpy as np


def phuong_phap_truot_ham_muc_tieu(obj_bai_toan, duong_bland=None, duong_dantzig=None):
    EPS = 1e-5

    # 1. Thu thập tất cả các đường thẳng
    duong_thang = []
    for he_so, dau, vp in obj_bai_toan.cac_rang_buoc:
        duong_thang.append((he_so[0], he_so[1], vp))

    for idx, dau in enumerate(obj_bai_toan.dau_cua_bien):
        if dau in [">=0", "<=0"]:
            if idx == 0:
                duong_thang.append((1, 0, 0))  # Phương trình x1 = 0 (Trục x2)
            else:
                duong_thang.append((0, 1, 0))  # Phương trình x2 = 0 (Trục x1)

    # Biên ảo để kiểm tra miền không giới nội
    gia_tri_lon_nhat = max([abs(vp) for he_so, dau, vp in obj_bai_toan.cac_rang_buoc])
    BIEN = max(gia_tri_lon_nhat * 2, 10)
    duong_thang.extend(
        [(1, 0, BIEN), (1, 0, -BIEN), (0, 1, BIEN), (0, 1, -BIEN)]
    )

    # 2. Tìm tất cả giao điểm
    giao_diem_all = []
    n = len(duong_thang)
    for i in range(n):
        for j in range(i + 1, n):
            a1, b1, c1 = duong_thang[i]
            a2, b2, c2 = duong_thang[j]
            D = a1 * b2 - a2 * b1
            if abs(D) > EPS:
                x1 = (c1 * b2 - c2 * b1) / D
                x2 = (a1 * c2 - a2 * c1) / D
                giao_diem_all.append((x1, x2))

    # 3. Lọc đỉnh hợp lệ 
    dinh_mien_nghiem = []
    for x1, x2 in giao_diem_all:
        hop_le = True
        for idx, dau in enumerate(obj_bai_toan.dau_cua_bien):
            val = x1 if idx == 0 else x2
            if dau == ">=0" and val < -EPS:
                hop_le = False
            if dau == "<=0" and val > EPS:
                hop_le = False

        for he_so, dau, vp in obj_bai_toan.cac_rang_buoc:
            tinh_toan = he_so[0] * x1 + he_so[1] * x2
            if dau == "<=" and tinh_toan > vp + EPS:
                hop_le = False
            if dau == ">=" and tinh_toan < vp - EPS:
                hop_le = False
            if dau == "=" and abs(tinh_toan - vp) > EPS:
                hop_le = False

        if abs(x1) > BIEN + EPS or abs(x2) > BIEN + EPS:
            hop_le = False

        if hop_le:
            
            da_ton_tai = False
            for dinh in dinh_mien_nghiem:
                if abs(x1 - dinh[0]) < EPS and abs(x2 - dinh[1]) < EPS:
                    da_ton_tai = True
                    break
            if not da_ton_tai:
                dinh_mien_nghiem.append((x1, x2))

    # 4. Chuẩn bị đồ thị & Tính vùng vẽ chuẩn xác
    fig, ax = plt.subplots(figsize=(8,8))
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_aspect(
        "equal"
    )  # Đảm bảo trục 1:1 để vector pháp tuyến vuông góc chuẩn học
 
    dinh_thuc_te = [
        p for p in dinh_mien_nghiem if abs(p[0]) < 900 and abs(p[1]) < 900
    ]

    if len(dinh_thuc_te) > 0:
    pts_thuc_te = np.array(dinh_thuc_te)
    min_bnd = np.min(pts_thuc_te, axis=0)
    max_bnd = np.max(pts_thuc_te, axis=0)
    
    # Tính pad và đảm bảo nó luôn có giá trị tối thiểu (ví dụ: ít nhất là 2 đơn vị)
    pad = np.maximum((max_bnd - min_bnd) * 0.2, 2.0) 
    
    vung_ve = [min_bnd[0] - pad[0], max_bnd[0] + pad[0], 
               min_bnd[1] - pad[1], max_bnd[1] + pad[1]]

    # 5. Vẽ các đường ràng buộc
    x_vals = np.linspace(vung_ve[0], vung_ve[1], 100)
    for idx, (he_so, dau, vp) in enumerate(obj_bai_toan.cac_rang_buoc):
        a, b = he_so[0], he_so[1]
        mau_bien = "#4A5568"
        if b != 0:
            y_vals = (vp - a * x_vals) / b
            ax.plot(x_vals, y_vals, color=mau_bien, linewidth=1.2, alpha=0.6, 
                    label=f" {idx+1}: {a}x1 + {b}x2 {dau} {vp}")
        else:
            ax.axvline(vp / a, color=mau_bien, linewidth=1.2, alpha=0.6, 
                       label=f" {idx+1}: {a}x1 {dau} {vp}")
    # 6. Tô màu miền nghiệm 
    if len(dinh_mien_nghiem) >= 3:
        dinh_de_ve = []
        for x1, x2 in dinh_mien_nghiem:
            cx = np.clip(x1, vung_ve[0] - 1, vung_ve[1] + 1)
            cy = np.clip(x2, vung_ve[2] - 1, vung_ve[3] + 1)
            dinh_de_ve.append((cx, cy))

        pts = np.array(dinh_de_ve)
        tam = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - tam[1], pts[:, 0] - tam[0])
        pts_sorted = pts[np.argsort(angles)]
        ax.fill(
            pts_sorted[:, 0],
            pts_sorted[:, 1],
            facecolor="green",
            alpha=0.15,
            label="Miền nghiệm D",
        )

    # =====================================================================
    # 7. MÔ PHỎNG SỰ TRƯỢT CỦA HÀM MỤC TIÊU TỪ HAI ĐẦU VÔ CÙNG
    # =====================================================================
    c1, c2 = obj_bai_toan.he_so_f_goc[0], obj_bai_toan.he_so_f_goc[1]
    loai = obj_bai_toan.loai_muc_tieu.lower()
    norm = np.sqrt(c1**2 + c2**2)

    if norm > 0:
        # Định vị tọa độ tâm của vùng vẽ để làm mốc tính toán giá trị Z từ xa
        tamt_x = (vung_ve[0] + vung_ve[1]) / 2
        tamt_y = (vung_ve[2] + vung_ve[3]) / 2
        z_trung_tam = c1 * tamt_x + c2 * tamt_y
        
        # Tạo khoảng cách "ở rất xa" dựa trên kích thước vùng vẽ
        tam_quet = max(vung_ve[1] - vung_ve[0], vung_ve[3] - vung_ve[2]) * 0.8
        delta_z = norm * tam_quet
        
        # Tính giá trị Z đại diện cho phía "âm vô cùng" và "dương vô cùng" trong tầm nhìn
        z_phia_am = z_trung_tam - delta_z
        z_phia_duong = z_trung_tam + delta_z

        # --- HÀM VẼ ĐƯỜNG THẲNG Z VÀ MŨI TÊN CHỈ HƯỚNG QUÉT ---
        # Đã đồng bộ net_ve mặc định là nét đứt "--" cho cả hai đường
        def ve_duong_z_va_huong_truot(z_val, label_name, color_line, do_dam_alpha, huong_mui_ten):
            # Vẽ đường thẳng hàm mục tiêu tại giá trị z_val với nét đứt "--" đồng nhất
            if c2 != 0:
                ax.plot(x_vals, (z_val - c1 * x_vals) / c2, color=color_line, 
                        linestyle="--", linewidth=1.2, alpha=do_dam_alpha, label=label_name)
                # Lấy một điểm nằm trên đường thẳng ở khu vực trung tâm để đặt mũi tên
                pt_arrow_x = tamt_x - (c1 * (z_val - c1 * tamt_x - c2 * tamt_y)) / (norm**2)
                pt_arrow_y = (z_val - c1 * pt_arrow_x) / c2
            else:
                ax.axvline(z_val / c1, color=color_line, 
                           linestyle="--", linewidth=1.2, alpha=do_dam_alpha, label=label_name)
                pt_arrow_x = z_val / c1
                pt_arrow_y = tamt_y

            # Tính toán vector hướng của mũi tên nhỏ dựa trên loại bài toán (Max/Min)
            dx = (c1 / norm) * 0.6 * huong_mui_ten
            dy = (c2 / norm) * 0.6 * huong_mui_ten
            
            # Vẽ mũi tên nhỏ tinh tế bám theo độ mờ alpha của đường thẳng
            ax.annotate("", xy=(pt_arrow_x + dx, pt_arrow_y + dy), xytext=(pt_arrow_x, pt_arrow_y),
                        arrowprops=dict(arrowstyle="->", color=color_line, alpha=do_dam_alpha, lw=1.5, mutation_scale=10))

        # --- TIẾN HÀNH VẼ DỰA TRÊN BẢN CHẤT BÀI TOÁN ---
        # Xác định hướng mũi tên: Bài toán MAX trượt theo hướng tăng Z (1), MIN trượt theo hướng giảm Z (-1)
        chieu_truot = 1 if loai == "max" else -1

        # Vẽ đường xuất phát (Z từ rất xa chuẩn bị quét vào miền nghiệm): Màu xám rõ hơn (alpha=0.7)
        z_start = z_phia_am if loai == "max" else z_phia_duong
        ve_duong_z_va_huong_truot(z_start, "Z từ vô cùng vào", "#718096", 0.7, chieu_truot)

        # Vẽ đường đi khuất (Z đã vượt qua nghiệm tối ưu chạy đi): Màu xám nhạt mờ hơn (alpha=0.35)
        z_end = z_phia_duong if loai == "max" else z_phia_am
        ve_duong_z_va_huong_truot(z_end, "Z đi ra vô cùng", "#718096", 0.35, chieu_truot)
    # 8. BIỆN LUẬN NGHIỆM 
    loai = obj_bai_toan.loai_muc_tieu.lower()

    if len(dinh_mien_nghiem) == 0:
        print("\n Bài toán vô nghiệm do miền chấp nhận được là rỗng")
        z_inf_text = "+inf" if loai == "min" else "-inf"
        print(f"--> Giá trị tối ưu: {loai.upper()}(Z) = {z_inf_text}")
        ax.text(
            np.mean(vung_ve[:2]),
            np.mean(vung_ve[2:]),
            f"Miền chấp nhận rỗng\n{loai.upper()}(Z) = {z_inf_text}",
            color="red",
            fontsize=11,
            ha="center",
            va="center",
            weight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="red"),
        )
    else:
        gia_tri_z = np.array([c1 * pt[0] + c2 * pt[1] for pt in dinh_mien_nghiem])
        z_opt = np.min(gia_tri_z) if loai == "min" else np.max(gia_tri_z)

        cac_idx_toi_uu = np.where(np.isclose(gia_tri_z, z_opt, atol=EPS))[0]
        dinh_toi_uu_list = [dinh_mien_nghiem[i] for i in cac_idx_toi_uu]

        NGUONG_CHECK = BIEN * 0.95
        chay_ra_vo_cung = any(
            abs(pt[0]) >= NGUONG_CHECK or abs(pt[1]) >= NGUONG_CHECK
            for pt in dinh_toi_uu_list
        )

        # --- TH KHÔNG GIỚI NỘI ---
        if chay_ra_vo_cung:
            print("\n Bài toán không giới nội")
            z_inf_text = "-inf" if loai == "min" else "+inf"
            print(f"--> Giá trị tối ưu: {loai.upper()}(Z) = {z_inf_text}")
            ax.text(
                np.mean(vung_ve[:2]),
                np.mean(vung_ve[2:]),
                f"Bài toán không giới nội\n{loai.upper()}(Z) = {z_inf_text}",
                color="orange",
                fontsize=11,
                ha="center",
                va="center",
                weight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="orange"),
            )

        else:
            # Lọc bỏ các đỉnh tối ưu thực sự trùng nhau về mặt giá trị
            dinh_toi_uu_unique = []
            for d in dinh_toi_uu_list:
                if not any(
                    abs(d[0] - u[0]) < EPS and abs(d[1] - u[1]) < EPS
                    for u in dinh_toi_uu_unique
                ):
                    dinh_toi_uu_unique.append(d)

            # --- TH VÔ SỐ NGHIỆM (Khi có từ 2 đỉnh phân biệt trở lên cùng tối ưu) ---
            if len(dinh_toi_uu_unique) >= 2:
                print("\n Bài toán có vô số nghiệm")
                print(f"--> Giá trị tối ưu: {loai.upper()}(Z) = {z_opt:.2f}")
                print("--> Nghiệm tối ưu là đoạn thẳng nối các đỉnh:")
                for i, dinh in enumerate(dinh_toi_uu_unique):
                    print(f"    Đỉnh {i+1}: ({dinh[0]:.2f}, {dinh[1]:.2f})")

                pt1, pt2 = dinh_toi_uu_unique[0], dinh_toi_uu_unique[1]
                print(
                    f"--> Biểu thức đoạn nghiệm: S = {{ λ*({pt1[0]:.2f}, {pt1[1]:.2f}) + (1-λ)*({pt2[0]:.2f}, {pt2[1]:.2f}) | 0 <= λ <= 1 }}"
                )

                pts_opt = np.array(dinh_toi_uu_unique)
                ax.plot(
                    pts_opt[:, 0],
                    pts_opt[:, 1],
                    color="#E53E3E",
                    linestyle="-",
                    linewidth=5,
                    zorder=10,
                    label="Đoạn nghiệm tối ưu",
                )
                ax.scatter(
                    pts_opt[:, 0],
                    pts_opt[:, 1],
                    color="#E53E3E",
                    s=120,
                    zorder=11,
                    edgecolors="black",
                )

            # --- TH NGHIỆM DUY NHẤT (Chỉ còn duy nhất 1 đỉnh sau khi lọc trùng) ---
            else:
                x1_opt, x2_opt = dinh_toi_uu_unique[0]
                print("\n Bài toán có nghiệm duy nhất")
                print(f"--> Nghiệm tối ưu là: x*= ({x1_opt:.2f}, {x2_opt:.2f})")
                print(f"--> Giá trị tối ưu: {loai.upper()}(Z) = {z_opt:.2f}")

                ax.scatter(
                    x1_opt,
                    x2_opt,
                    color="red",
                    edgecolors="black",
                    s=100,                 # Điểm tròn to rõ ràng, không thể bị bỏ sót
                    zorder=10,             # Nằm trên cùng đồ thị
                    label=f"Tối ưu duy nhất ({x1_opt:.2f}, {x2_opt:.2f})",
                )

                if c2 != 0:
                    ax.plot(
                        x_vals,
                        (z_opt - c1 * x_vals) / c2,
                        color="purple",
                        linestyle="-.",
                        linewidth=2,
                        zorder=8,          # Nằm ngay dưới điểm tối ưu một chút
                        label=f"Đường Z_opt = {z_opt:.2f}",
                    )
                else:
                    ax.axvline(
                        z_opt / c1,
                        color="purple",
                        linestyle="-.",
                        linewidth=2,
                        zorder=8,
                        label=f"Đường Z_opt = {z_opt:.2f}",
                    )

    
    # =====================================================================
    # 9. VẼ ĐƯỜNG ĐI THUẬT TOÁN ĐƠN HÌNH (LINH HOẠT THỜI GIAN THỰC)
    # =====================================================================
    def ve_lo_trinh_don_hinh(duong_di, mau_sac, ten_nhan, vi_tri_lech=0, hinh_nut="o"):
        if not duong_di or len(duong_di) == 0:
            return
        
        x_coords, y_coords = zip(*duong_di)
        
        # Tạo độ lệch nhẹ tránh đè đường nếu vẽ chung cả 2 thuật toán
        x_coords = [x + vi_tri_lech for x in x_coords]
        y_coords = [y + vi_tri_lech for y in y_coords]
        
        # Vẽ đoạn thẳng lộ trình
        ax.plot(x_coords, y_coords, color=mau_sac, linestyle="-", linewidth=2.5, zorder=5, label=ten_nhan)
        
        # Vẽ các điểm nút lặp (Bland hình tròn, Dantzig hình tam giác để dễ phân biệt)
        ax.scatter(x_coords, y_coords, color="black", marker=hinh_nut, s=35, zorder=6)
        
        # Vẽ mũi tên hướng đi
        for i in range(len(duong_di) - 1):
            x1, y1 = x_coords[i], y_coords[i]
            x2, y2 = x_coords[i+1], y_coords[i+1]
            if abs(x1 - x2) > EPS or abs(y1 - y2) > EPS:
                ax.annotate(
                    "", 
                    xy=(x2, y2), 
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="->", 
                        color=mau_sac, 
                        lw=2, 
                        mutation_scale=12,
                        connectionstyle="arc3"
                    )
                )

    # --- ĐIỀU KHIỂN LOGIC HIỂN THỊ TỰ ĐỘNG (Ngang hàng với hàm ve_lo_trinh_don_hinh) ---
    
    # 1. Vẽ đường Bland (Luôn luôn vẽ nếu có dữ liệu - Áp dụng cho cả Hai pha và Đơn hình thường)
    if duong_bland:
        # Nếu có cả Dantzig thì dịch chuyển một chút (-0.05), nếu đi một mình (Hai pha) thì giữ nguyên gốc (0)
        lech_bland = -0.05 if duong_dantzig else 0
        ve_lo_trinh_don_hinh(duong_bland, mau_sac="#1A365D", ten_nhan="Lộ trình Bland", vi_tri_lech=lech_bland, hinh_nut="o")
        
    # 2. Vẽ đường Dantzig (Chỉ vẽ khi bài toán b_i > 0 và người dùng chủ động truyền vào để so sánh)
    if duong_dantzig:
        ve_lo_trinh_don_hinh(duong_dantzig, mau_sac="#C53030", ten_nhan="Lộ trình Dantzig", vi_tri_lech=0.05, hinh_nut="^")

    # Đánh dấu Đỉnh khởi đầu và Đích tối ưu chung cho đồ thị (Lấy từ điểm đầu/cuối của đường Bland làm chuẩn)
    if duong_bland:
        ax.scatter(duong_bland[0][0], duong_bland[0][1], color="lime", edgecolor="black", s=100, zorder=7, label="Đỉnh khởi đầu")
        ax.scatter(duong_bland[-1][0], duong_bland[-1][1], color="gold", edgecolor="black", marker="*", s=180, zorder=7, label="Đích Đơn hình")
    
    # FIX: Đưa các dòng cấu hình này ra ngoài hàm con ve_lo_trinh_don_hinh
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle=":")
    
    plt.show()