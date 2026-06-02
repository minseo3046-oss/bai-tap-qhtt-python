import matplotlib.pyplot as plt
import numpy as np


def phuong_phap_truot_ham_muc_tieu(obj_bai_toan):
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
    BIEN = 1000
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
    fig, ax = plt.subplots(figsize=(7, 7))
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
        vung_ve = [min_bnd[0] - 2, max_bnd[0] + 5, min_bnd[1] - 2, max_bnd[1] + 5]
    else:
        vung_ve = [-2, 12, -2, 12]

    ax.set_xlim(vung_ve[0], vung_ve[1])
    ax.set_ylim(vung_ve[2], vung_ve[3])

    # 5. Vẽ các đường ràng buộc
    x_vals = np.linspace(vung_ve[0], vung_ve[1], 100)
    for he_so, dau, vp in obj_bai_toan.cac_rang_buoc:
        a, b = he_so[0], he_so[1]
        if b != 0:
            y_vals = (vp - a * x_vals) / b
            ax.plot(x_vals, y_vals, label=f"{a}x1 + {b}x2 {dau} {vp}")
        else:
            ax.axvline(vp / a, label=f"{a}x1 {dau} {vp}")

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

    # 7. Vẽ đường Z = 1 và Vector pháp tuyến hướng tăng Z
    c1, c2 = obj_bai_toan.he_so_f_goc[0], obj_bai_toan.he_so_f_goc[1]
    if c2 != 0:
        ax.plot(x_vals, (1 - c1 * x_vals) / c2, color="blue", label="Z = 1")
    else:
        ax.axvline(1 / c1, color="blue", label="Z = 1")

    norm = np.sqrt(c1**2 + c2**2)
    if norm > 0:
        ax.arrow(
            0,
            0,
            (c1 / norm) * 1.5,
            (c2 / norm) * 1.5,
            head_width=0.2,
            fc="red",
            ec="red",
            label="Hướng tăng Z",
        )

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

        NGUONG_CHECK = 990
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
                    color="red",
                    linestyle="--",
                    linewidth=4,
                    label="Đoạn nghiệm tối ưu",
                )
                ax.scatter(
                    pts_opt[:, 0],
                    pts_opt[:, 1],
                    color="red",
                    s=100,
                    zorder=5,
                    edgecolors="black",
                )

            # --- TH NGHIỆM DUY NHẤT (Chỉ còn duy nhất 1 đỉnh sau khi lọc trùng) ---
            else:
                x1_opt, x2_opt = dinh_toi_uu_unique[0]
                print("\n Bài toán có nghiệm duy nhất")
                print(f"--> Nghiệm tối ưu là: x*= ({x1_opt:.2f}, {x2_opt:.2f})")
                print(f"--> Giá trị tối ưu: {loai.upper()}(Z) = {z_opt:.2f}")

                ax.plot(
                    x1_opt,
                    x2_opt,
                    "ro",
                    markersize=8,
                    label=f"Tối ưu duy nhất ({x1_opt:.2f}, {x2_opt:.2f})",
                )

                if c2 != 0:
                    ax.plot(
                        x_vals,
                        (z_opt - c1 * x_vals) / c2,
                        color="purple",
                        linestyle="-.",
                        linewidth=2,
                        label=f"Đường Z_opt = {z_opt:.2f}",
                    )
                else:
                    ax.axvline(
                        z_opt / c1,
                        color="purple",
                        linestyle="-.",
                        linewidth=2,
                        label=f"Đường Z_opt = {z_opt:.2f}",
                    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":")
    plt.show()
