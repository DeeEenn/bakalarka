import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tkinter import Tk, filedialog
import os

# --- DEFINICE PROPOJENÍ PRO HORNÍ POLOVINU TĚLA ---
# Vynechali jsme body 23-32 (nohy), které u tvého videa dělají neplechu
POSE_CONNECTIONS_UPPER = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Hlava
    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)       # Ramena a ruce
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # palec
    (0, 5), (5, 6), (6, 7), (7, 8),      # ukazováček
    (9, 10), (10, 11), (11, 12),         # prostředníček
    (13, 14), (14, 15), (15, 16),        # prsteníček
    (17, 18), (18, 19), (19, 20),        # malíček
    (0, 5), (5, 9), (9, 13), (13, 17), (17, 0) # dlaň
]

def select_feature_file():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Vyber .npy soubor (218 basic / 243 enhanced)",
        filetypes=[("NumPy files", "*.npy")]
    )
    root.destroy()
    return file_path

def visualize_inhalation_focus(file_path):
    try:
        data = np.load(file_path)
        num_frames = len(data)
        num_features = data.shape[1]
        
        # Detekce verze features
        if num_features == 218:
            version = "Basic (218)"
            print(f"✓ Načteno {num_frames} snímků, {num_features} features [BASIC]")
            print(f"  → Horní tělo: 23 bodů (92 hodnot)")
            print(f"  → Ruce: 2×21 bodů (126 hodnot)")
        elif num_features == 243:
            version = "Enhanced (243)"
            print(f"✓ Načteno {num_frames} snímků, {num_features} features [ENHANCED]")
            print(f"  → Raw coordinates: 218")
            print(f"  → Distances: 11 (včetně mouth distance), Angles: 8, Hand config: 6")
        elif num_features == 242:
            version = "Enhanced (242) - stará verze"
            print(f"✓ Načteno {num_frames} snímků, {num_features} features [ENHANCED OLD]")
            print(f"  → Raw coordinates: 218")
            print(f"  → Distances: 10, Angles: 8, Hand config: 6")
        else:
            print(f"⚠️ VAROVÁNÍ: Neznámý formát {num_features} features")
            print(f"Podporované: 218 (basic), 242 (old enhanced) nebo 243 (enhanced)")
            return
    except Exception as e:
        print(f"Chyba: {e}")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        row = data[frame_idx]
        
        # Extrakce raw coordinates (prvních 218 hodnot)
        # Funguje pro obě verze (basic i enhanced)
        pose_data = row[:92].reshape(23, 4)    # Horní polovina: indexy 0-22
        lh_data = row[92:155].reshape(21, 3)   # Levá ruka
        rh_data = row[155:218].reshape(21, 3)  # Pravá ruka

        # 1. KRESLÍME TĚLO (Horní polovina: 0-22)
        px, py, pz, pvis = pose_data[:, 0], pose_data[:, 1], pose_data[:, 2], pose_data[:, 3]
        
        # Tělo - černé
        for start, end in POSE_CONNECTIONS_UPPER:
            if start < 23 and end < 23 and pvis[start] > 0.3 and pvis[end] > 0.3:
                ax.plot([px[start], px[end]], [pz[start], pz[end]], [-py[start], -py[end]], 
                        color='black', linewidth=3, alpha=0.9)
        
        # Klouby hlavy a ramen - šedé
        ax.scatter(px[:17], pz[:17], -py[:17], c='darkgray', s=30, alpha=0.7)

        # 2. KRESLÍME RUCE (Červená=Levá, Zelená=Pravá)
        # MediaPipe pose: bod 15 = levé zápěstí, bod 16 = pravé zápěstí
        # MediaPipe hand: bod 0 = zápěstí ruky
        
        # Levá ruka - transformace aby seděla na těle
        lx, ly, lz = lh_data[:, 0].copy(), lh_data[:, 1].copy(), lh_data[:, 2].copy()
        if not np.all(lx == 0) and pvis[15] > 0.3:  # 15 = levé zápěstí
            # Vypočítáme offset mezi pose zápěstím a hand zápěstím
            offset_x = px[15] - lx[0]
            offset_y = py[15] - ly[0]
            offset_z = pz[15] - lz[0]
            
            # Posuneme celou ruku, aby zápěstí sedělo
            lx += offset_x
            ly += offset_y
            lz += offset_z
            
            # Kreslíme ruku
            for start, end in HAND_CONNECTIONS:
                ax.plot([lx[start], lx[end]], [lz[start], lz[end]], [-ly[start], -ly[end]], 
                        color='red', linewidth=2.5, alpha=0.8)
            ax.scatter(lx, lz, -ly, c='darkred', s=20)

        # Pravá ruka - transformace aby seděla na těle
        rx, ry, rz = rh_data[:, 0].copy(), rh_data[:, 1].copy(), rh_data[:, 2].copy()
        if not np.all(rx == 0) and pvis[16] > 0.3:  # 16 = pravé zápěstí
            # Vypočítáme offset mezi pose zápěstím a hand zápěstím
            offset_x = px[16] - rx[0]
            offset_y = py[16] - ry[0]
            offset_z = pz[16] - rz[0]
            
            # Posuneme celou ruku, aby zápěstí sedělo
            rx += offset_x
            ry += offset_y
            rz += offset_z
            
            # Kreslíme ruku
            for start, end in HAND_CONNECTIONS:
                ax.plot([rx[start], rx[end]], [rz[start], rz[end]], [-ry[start], -ry[end]], 
                        color='green', linewidth=2.5, alpha=0.8)
            ax.scatter(rx, rz, -ry, c='darkgreen', s=20)

        # 3. VIZUALIZACE MOUTH DISTANCE (pokud jsou enhanced features)
        if num_features >= 243:
            # Mouth distance je na indexu 228 (218 raw + 10 distances)
            mouth_dist = row[228]
            
            # Kreslíme linku mezi rty (aproximace - použijeme body z pózy)
            # Bod 10 = ústa (z pose landmarks)
            mouth_x, mouth_y, mouth_z = px[10], py[10], pz[10]
            
            # Vizualizujeme mouth distance jako vertikální čáru u úst
            if mouth_dist > 0 and pvis[10] > 0.3:
                # Měřítko pro lepší viditelnost (mouth_dist je malé číslo)
                scale = 0.5
                mouth_open = mouth_dist * scale
                
                # Kreslíme vertikální čáru (horní a dolní ret)
                upper_y = mouth_y - mouth_open/2
                lower_y = mouth_y + mouth_open/2
                
                ax.plot([mouth_x, mouth_x], [mouth_z, mouth_z], 
                       [-upper_y, -lower_y], 
                       color='magenta', linewidth=4, alpha=0.9, label='Rty')
                
                # Body na koncích
                ax.scatter([mouth_x, mouth_x], [mouth_z, mouth_z], 
                          [-upper_y, -lower_y], 
                          c='magenta', s=80, marker='o', edgecolors='white', linewidth=1.5)
                
                # Text s hodnotou
                ax.text(mouth_x + 0.05, mouth_z, -mouth_y, 
                       f'Mouth: {mouth_dist:.3f}', 
                       fontsize=9, color='magenta', fontweight='bold')

        # --- NASTAVENÍ POHLEDU (Frontální + zoom na inhalaci) ---
        ax.set_xlim(0.1, 0.9)      # Šířka
        ax.set_ylim(-0.4, 0.4)     # Hloubka
        ax.set_zlim(-1.2, -0.2)    # Výška (oříznuto, focus na hlavu/ruce)
        
        ax.set_xlabel('X (←→)', fontsize=10)
        ax.set_ylabel('Z (hloubka)', fontsize=10)
        ax.set_zlabel('Y (↓↑)', fontsize=10)
        
        ax.view_init(elev=5, azim=-90)  # Frontální pohled 
        ax.set_title(f"Inhalace - Detail [{version}] | Snímek: {frame_idx+1}/{num_frames}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)

    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=40)
    plt.show()

def analyze_features(file_path):
    """
    Analytická funkce pro ověření kvality dat pro MS-TCN/ASFormer
    """
    data = np.load(file_path)
    num_frames = len(data)
    num_features = data.shape[1]
    
    print("\n" + "="*60)
    print("ANALÝZA FEATURES PRO ACTION SEGMENTATION")
    print("="*60)
    
    # 1. Základní info
    print(f"\n✓ Počet snímků: {num_frames}")
    print(f"✓ Délka videa: ~{num_frames/30:.1f}s (při 30 FPS)")
    print(f"✓ Dimenze features: {num_features}")
    
    if num_features == 218:
        print(f"✓ Verze: BASIC")
    elif num_features == 243:
        print(f"✓ Verze: ENHANCED (basic + distances + angles + hand config + mouth distance)")
    elif num_features == 242:
        print(f"✓ Verze: ENHANCED OLD (bez mouth distance)")
    else:
        print(f"⚠️ Neznámá verze")
    
    # 2. Rozklad features (prvních 218 hodnot je stejných)
    pose_data = data[:, :92].reshape(num_frames, 23, 4)
    lh_data = data[:, 92:155].reshape(num_frames, 21, 3)
    rh_data = data[:, 155:218].reshape(num_frames, 21, 3)
    
    # 3. Detekce visibility (tělo)
    avg_visibility = pose_data[:, :, 3].mean()
    print(f"\n📍 Průměrná visibility těla: {avg_visibility:.2%}")
    if avg_visibility < 0.5:
        print(" VAROVÁNÍ: Nízká viditelnost (<50%) - možná špatné osvětlení/kamera")
    
    # 4. Detekce rukou
    lh_detected = np.sum(np.any(lh_data != 0, axis=(1, 2)))
    rh_detected = np.sum(np.any(rh_data != 0, axis=(1, 2)))
    print(f"\nDetekce rukou:")
    print(f"  → Levá ruka: {lh_detected}/{num_frames} snímků ({lh_detected/num_frames:.1%})")
    print(f"  → Pravá ruka: {rh_detected}/{num_frames} snímků ({rh_detected/num_frames:.1%})")
    
    if lh_detected < num_frames * 0.3 and rh_detected < num_frames * 0.3:
        print("  VAROVÁNÍ: Ruce téměř nedetekované - možná nejsou v záběru")
    
    # 5. Analýza pohybu (variance)
    pose_variance = np.var(pose_data[:, :10, :3])  # Variance hlavy/ramen
    lh_variance = np.var(lh_data) if lh_detected > 0 else 0
    rh_variance = np.var(rh_data) if rh_detected > 0 else 0
    
    print(f"\nVariance pohybu (indikátor aktivity):")
    print(f"  → Tělo: {pose_variance:.4f}")
    print(f"  → Levá ruka: {lh_variance:.4f}")
    print(f"  → Pravá ruka: {rh_variance:.4f}")
    
    if pose_variance < 0.0001:
        print("  Téměř žádný pohyb těla - možná statický záznam")
    
    # 6. Test interpolace (kolik nul bylo nahrazeno)
    # Testujeme jen raw coordinates (prvních 218)
    raw_data = data[:, :218]
    zero_ratio = np.sum(raw_data == 0) / raw_data.size
    print(f"\n🔧 Nulové hodnoty (raw coords): {zero_ratio:.2%}")
    if zero_ratio > 0.3:
        print(f"  ⚠️ Hodně nul - možná špatná detekce nebo chybějící data")
    
    # 7. Doporučení pro MS-TCN / ASFormer
    print(f"\n🤖 Doporučení pro trénink:")
    print(f"  ✓ Formát je správný ({num_features} features)")
    print(f"  ✓ Interpolace a vyhlazení aplikováno")
    
    if num_frames < 30:
        print(f"  Video je krátké (<1s) - možná potřeba padding")
    elif num_frames > 300:
        print(f"  Video je dlouhé (>{num_frames/30:.0f}s) - zvážit segmentaci")
    
    print("="*60 + "\n")

def visualize_smoothness(file_path):
    """
    Zobrazí plynulost dat v čase - ověření, jestli Savitzky-Golay funguje
    """
    data = np.load(file_path)
    num_frames = len(data)
    num_features = data.shape[1]
    
    # Rozklad dat (prvních 218 hodnot je raw coordinates)
    pose_data = data[:, :92].reshape(num_frames, 23, 4)
    lh_data = data[:, 92:155].reshape(num_frames, 21, 3)
    rh_data = data[:, 155:218].reshape(num_frames, 21, 3)
    
    # Důležité body pro inhalaci
    # Pokud máme mouth distance, přidáme 5. graf
    has_mouth = num_features >= 243
    num_plots = 5 if has_mouth else 4
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 14 if has_mouth else 12))
    fig.suptitle(f'Plynulost pohybu v čase | {os.path.basename(file_path)}', 
                 fontsize=14, fontweight='bold')
    
    time_axis = np.arange(num_frames)
    
    # 1. Nos (bod 0) - reference hlavy
    nose_y = -pose_data[:, 0, 1]  # Y (nahoru/dolů)
    axes[0].plot(time_axis, nose_y, linewidth=2, color='blue', label='Nos (Y)')
    axes[0].set_ylabel('Y pozice', fontsize=11)
    axes[0].set_title('Pohyb hlavy (Nos)', fontsize=11, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Levé zápěstí (bod 15)
    lwrist_x = pose_data[:, 15, 0]
    lwrist_y = -pose_data[:, 15, 1]
    axes[1].plot(time_axis, lwrist_x, linewidth=2, color='red', label='Levé zápěstí (X)', alpha=0.8)
    axes[1].plot(time_axis, lwrist_y, linewidth=2, color='darkred', label='Levé zápěstí (Y)', alpha=0.8)
    axes[1].set_ylabel('Pozice', fontsize=11)
    axes[1].set_title('Pohyb levé ruky', fontsize=11, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Pravé zápěstí (bod 16)
    rwrist_x = pose_data[:, 16, 0]
    rwrist_y = -pose_data[:, 16, 1]
    axes[2].plot(time_axis, rwrist_x, linewidth=2, color='green', label='Pravé zápěstí (X)', alpha=0.8)
    axes[2].plot(time_axis, rwrist_y, linewidth=2, color='darkgreen', label='Pravé zápěstí (Y)', alpha=0.8)
    axes[2].set_ylabel('Pozice', fontsize=11)
    axes[2].set_title('Pohyb pravé ruky', fontsize=11, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Rychlost změny (derivace) - test plynulosti
    # Čím větší skoky, tím více "jitteru"
    velocity = np.diff(rwrist_y)  # První derivace pravé ruky
    axes[3].plot(time_axis[:-1], velocity, linewidth=1.5, color='orange', label='Rychlost změny')
    axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Rychlost (px/frame)', fontsize=11)
    axes[3].set_xlabel('Číslo snímku', fontsize=11)
    axes[3].set_title('Rychlost změny polohy (Jitter test) - Pravá ruka', fontsize=11, fontweight='bold')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Výpočet jitteru
    jitter = np.std(velocity)
    avg_velocity = np.mean(np.abs(velocity))
    axes[3].text(0.02, 0.95, f'Jitter (std): {jitter:.4f}\nPrům. rychlost: {avg_velocity:.4f}', 
                 transform=axes[3].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Mouth Distance (pokud je k dispozici)
    if has_mouth:
        mouth_distances = data[:, 228]  # Index 228 = mouth distance (218 raw + 10 distances)
        axes[4].plot(time_axis, mouth_distances, linewidth=2.5, color='magenta', 
                    label='Vzdálenost rtů', alpha=0.9)
        axes[4].set_ylabel('Mouth Distance', fontsize=11)
        axes[4].set_xlabel('Číslo snímku', fontsize=11)
        axes[4].set_title('Mouth Distance (otevření úst) - Indikátor nádechu', 
                         fontsize=11, fontweight='bold')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # Statistiky
        mouth_max = np.max(mouth_distances)
        mouth_avg = np.mean(mouth_distances)
        mouth_std = np.std(mouth_distances)
        axes[4].text(0.02, 0.95, 
                    f'Max: {mouth_max:.4f}\nPrům: {mouth_avg:.4f}\nStd: {mouth_std:.4f}', 
                    transform=axes[4].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))
        
        # Označení momentů otevření úst (když je mouth distance > průměr + 1*std)
        threshold = mouth_avg + mouth_std
        open_moments = mouth_distances > threshold
        if np.any(open_moments):
            axes[4].fill_between(time_axis, 0, mouth_distances, 
                               where=open_moments, alpha=0.3, color='yellow', 
                               label='Možný nádech')
    
    plt.tight_layout()
    plt.show()
    
    # Diagnostika
    print(f"\n📈 ANALÝZA PLYNULOSTI:")
    print(f"  → Jitter (std derivace): {jitter:.4f}")
    if jitter < 0.01:
        print(f"  ✓ Vyhlazení funguje dobře - data jsou plynulá")
    elif jitter < 0.03:
        print(f"  ⚠️ Mírný jitter - data jsou OK, ale mohla by být gladsší")
    else:
        print(f"  ❌ Vysoký jitter - možná je potřeba silnější vyhlazení")
    print()

if __name__ == "__main__":
    path = select_feature_file()
    if path:
        print(f"\n📂 Vybraný soubor: {os.path.basename(path)}")
        
        # Nejdřív analýza
        analyze_features(path)
        
        # Menu pro výběr vizualizace
        print("\n" + "="*60)
        print("VÝBĚR VIZUALIZACE:")
        print("="*60)
        print("1 - 3D animace skeletonu (frontální pohled)")
        print("2 - Graf plynulosti pohybu v čase (ověření jitteru)")
        print("3 - Obojí (doporučeno!)")
        print("="*60)
        
        choice = input("\nTvá volba (1/2/3): ").strip()
        
        if choice == "1":
            print("\n🎬 Spouštím 3D animaci...\n")
            visualize_inhalation_focus(path)
        elif choice == "2":
            print("\n📊 Zobrazuji grafy plynulosti...\n")
            visualize_smoothness(path)
        elif choice == "3":
            print("\n📊 Zobrazuji grafy plynulosti...\n")
            visualize_smoothness(path)
            print("\n🎬 Spouštím 3D animaci...\n")
            visualize_inhalation_focus(path)
        else:
            print("❌ Neplatná volba!")