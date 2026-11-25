import os
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import fitz

win_fonts = r"C:\Windows\Fonts"

# Register Arial or Segoe UI - let's use Segoe UI or Arial
# Segoe UI has beautiful, crisp, refined bold and regular weights that look finished and high quality.
pdfmetrics.registerFont(TTFont("ResumeFont", os.path.join(win_fonts, "segoeui.ttf")))
pdfmetrics.registerFont(TTFont("ResumeFont-Bold", os.path.join(win_fonts, "segoeuib.ttf")))
pdfmetrics.registerFont(TTFont("ResumeFont-Italic", os.path.join(win_fonts, "segoeuii.ttf")))

FONT_REG = "ResumeFont"
FONT_BOLD = "ResumeFont-Bold"
FONT_ITALIC = "ResumeFont-Italic"

PAGE_W, PAGE_H = letter # 612 x 792
X_LEFT = 28.8
X_RIGHT = 583.1

def py_to_c(y_top):
    # In original PDF, y is top-down (0 at top, 792 at bottom).
    # In ReportLab, text baseline is slightly below bbox top.
    # An offset of ~7.5-8.0pt from y_top gives exact baseline alignment.
    return PAGE_H - (y_top + 7.8)

def draw_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    
    # Lines helper
    def draw_line(y_top):
        c.setLineWidth(0.7)
        c.setStrokeColorRGB(0, 0, 0)
        c_y = PAGE_H - y_top
        c.line(X_LEFT, c_y, X_RIGHT, c_y)

    # 1. Header Name: y=14.00, size 20.0pt
    c.setFont(FONT_BOLD, 19.5)
    name = "LAKSHIT MALHOTRA"
    name_w = c.stringWidth(name, FONT_BOLD, 19.5)
    c.drawString((PAGE_W - name_w) / 2.0, PAGE_H - (14.0 + 15.5), name)
    
    # 2. Header Contact Line: y=35.10, size 9.0pt
    c.setFont(FONT_REG, 9.0)
    email = "lakshitmalhotra860@gmail.com"
    phone = "+91-7988635897"
    sep = " •  "
    
    contact_parts = [
        (email, "mailto:lakshitmalhotra860@gmail.com"),
        (sep, None),
        (phone, "tel:+917988635897"),
        (sep, None),
        ("LinkedIn", "https://www.linkedin.com/in/lakshitmalhotra"),
        (sep, None),
        ("GitHub", "https://github.com/lakshitmalhotra01"),
        (sep, None),
        ("LeetCode", "https://leetcode.com/u/lakshit-malhotra/")
    ]
    tot_w = sum(c.stringWidth(txt, FONT_REG, 9.0) for txt, _ in contact_parts)
    cur_x = (PAGE_W - tot_w) / 2.0
    c_y = py_to_c(35.10)
    
    for txt, url in contact_parts:
        w = c.stringWidth(txt, FONT_REG, 9.0)
        c.drawString(cur_x, c_y, txt)
        if url:
            c.linkURL(url, (cur_x, c_y - 2, cur_x + w, c_y + 9), relative=0)
        cur_x += w
        
    # Section Heading helper
    def draw_heading(title, y_top):
        c.setFont(FONT_BOLD, 11.0)
        c.drawString(X_LEFT, py_to_c(y_top), title)

    # OBJECTIVE
    draw_heading("OBJECTIVE", 52.10)
    draw_line(63.90)
    
    c.setFont(FONT_REG, 9.5)
    c.drawString(X_LEFT, py_to_c(69.20), "Pursuing a Computer Science undergraduate degree with strong foundations in machine learning, algorithms, and software engineering.")
    c.drawString(X_LEFT, py_to_c(80.80), "I am committed to developing robust technical expertise and applying my knowledge to solve real-world problems.")
    
    # EDUCATION
    draw_heading("EDUCATION", 96.90)
    draw_line(108.70)
    
    # Bennett University, Greater Noida (Bold on university heading as requested)
    c.setFont(FONT_BOLD, 9.5)
    c.drawString(X_LEFT, py_to_c(114.00), "Bennett University, Greater Noida")
    c.setFont(FONT_REG, 9.5)
    date1 = "2024 – 2028"
    d1_w = c.stringWidth(date1, FONT_REG, 9.5)
    c.drawString(X_RIGHT - d1_w, py_to_c(114.10), date1)
    
    c.setFont(FONT_REG, 9.5)
    c.drawString(X_LEFT, py_to_c(125.40), "B.Tech – Computer Science & Engineering | CGPA: 8.98/10")
    
    # Pratap Public School, Karnal, Haryana (Bold on school heading)
    c.setFont(FONT_BOLD, 9.5)
    c.drawString(X_LEFT, py_to_c(138.70), "Pratap Public School, Karnal, Haryana")
    c.setFont(FONT_REG, 9.5)
    date2 = "2022 – 2024"
    d2_w = c.stringWidth(date2, FONT_REG, 9.5)
    c.drawString(X_RIGHT - d2_w, py_to_c(138.80), date2)
    
    c.setFont(FONT_REG, 9.5)
    c.drawString(X_LEFT, py_to_c(150.10), "Class XII (CBSE): 90% | Class X (CBSE): 96.6%")
    
    # PROFESSIONAL EXPERIENCE
    draw_heading("PROFESSIONAL EXPERIENCE", 166.20)
    draw_line(178.00)
    
    # Internship 1: Software Developer Intern (Bold on internship heading as requested)
    c.setFont(FONT_BOLD, 9.4)
    c.drawString(X_LEFT, py_to_c(183.30), "Software Developer Intern — Centre for Railway Information Systems (CRIS), Technical Department")
    c.setFont(FONT_REG, 9.5)
    d3 = "04/08/2026 – 30/09/2026"
    d3_w = c.stringWidth(d3, FONT_REG, 9.5)
    c.drawString(X_RIGHT - d3_w, py_to_c(183.40), d3)
    
    # Bullets with original '-' (no dots)
    c.setFont(FONT_REG, 9.4)
    c.drawString(34.00, py_to_c(194.70), "- Contributed to the IR MARS Web Application as an integral part of the team, working on both backend and frontend development to")
    c.drawString(44.70, py_to_c(206.30), "deliver an efficient, user-friendly application.")
    
    c.drawString(34.00, py_to_c(218.70), "- Worked on the AI/ML component of the organisation's website, developing, evaluating, and refining an intelligent chatbot to assist")
    c.drawString(44.70, py_to_c(230.30), "users with queries related to railway applications and services.")
    
    c.drawString(34.00, py_to_c(242.70), "- Set up the project environment, mentored fellow new interns, and delivered an in-house presentation demonstrating software")
    c.drawString(44.70, py_to_c(254.30), "engineering principles and project outcomes.")
    
    # Internship 2: Research Intern (Bold on internship heading as requested)
    c.setFont(FONT_BOLD, 9.4)
    c.drawString(X_LEFT, py_to_c(267.20), "Research Intern — Bennett University")
    c.setFont(FONT_REG, 9.5)
    d4 = "Remote | 01/06/2026 – 30/07/2026"
    d4_w = c.stringWidth(d4, FONT_REG, 9.5)
    c.drawString(X_RIGHT - d4_w, py_to_c(267.30), d4)
    
    c.setFont(FONT_REG, 9.4)
    c.drawString(34.00, py_to_c(278.60), "- Carried out a faculty-guided research project on Computer Vision and AI/ML for intelligent mushroom growth monitoring, covering")
    c.drawString(44.70, py_to_c(290.20), "image segmentation, growth-stage classification, and yield prediction on real-world image data.")
    
    c.drawString(34.00, py_to_c(302.60), "- Formulated the problem, designed and ran experiments, and benchmarked 7 deep learning architectures, with the Swin Transformer")
    c.drawString(44.70, py_to_c(314.20), "achieving the best accuracy (~98%).")
    
    # PROJECTS
    draw_heading("PROJECTS", 329.90)
    draw_line(341.70)
    
    # Project 1: Satellite Optical Image Denoising (Bold on project heading, clickable link directly on heading)
    p1_title = "Satellite Optical Image Denoising using Deep Learning"
    c.setFont(FONT_BOLD, 9.8)
    c.drawString(X_LEFT, py_to_c(347.10), p1_title)
    p1_w = c.stringWidth(p1_title, FONT_BOLD, 9.8)
    p1_cy = py_to_c(347.10)
    c.linkURL("https://github.com/lakshitmalhotra01/Deep-SAR-Despeckling-Benchmark", (X_LEFT, p1_cy - 2, X_LEFT + p1_w, p1_cy + 10), relative=0)
    
    c.setFont(FONT_ITALIC, 9.5)
    c.drawString(X_LEFT, py_to_c(358.40), "Python · Deep Learning · Image Restoration · Residual U-Net · NAFNet · Restormer")
    
    c.setFont(FONT_REG, 9.4)
    c.drawString(34.00, py_to_c(371.40), "- Built a deep learning pipeline to restore 5,000 S2Looking RGB satellite images corrupted with synthetic Gamma multiplicative speckle")
    c.drawString(44.70, py_to_c(383.00), "noise (L=4), simulating SAR-style degradation on optical imagery.")
    
    c.drawString(34.00, py_to_c(395.40), "- Benchmarked 6 architectures (Residual, Attention & CBAM U-Net, DnCNN, NAFNet, Restormer) on PSNR and SSIM to identify the")
    c.drawString(44.70, py_to_c(407.00), "best-performing restoration model.")
    
    c.drawString(34.00, py_to_c(419.40), "- Achieved 37.21 dB PSNR and 98.89% pixel accuracy with Residual U-Net, trained using AMP, AdamW and a combined L1 + SSIM loss.")
    
    # Project 2: Smart Mushroom Farming (Bold on project heading, clickable link directly on heading)
    p2_title = "Smart Mushroom Farming — Faculty-Guided Research Project"
    c.setFont(FONT_BOLD, 9.8)
    c.drawString(X_LEFT, py_to_c(431.80), p2_title)
    p2_w = c.stringWidth(p2_title, FONT_BOLD, 9.8)
    p2_cy = py_to_c(431.80)
    c.linkURL("https://github.com/lakshitmalhotra01/smart-mushroom-farming-summer-internship-project-", (X_LEFT, p2_cy - 2, X_LEFT + p2_w, p2_cy + 10), relative=0)
    
    c.setFont(FONT_ITALIC, 9.5)
    c.drawString(X_LEFT, py_to_c(443.10), "PyTorch · timm · Swin Transformer · Albumentations · Grad-CAM · Streamlit")
    
    c.setFont(FONT_REG, 9.4)
    c.drawString(34.00, py_to_c(456.10), "- Developed an end-to-end deep learning pipeline for automated mushroom growth-stage monitoring, covering dataset preparation,")
    c.drawString(44.70, py_to_c(467.70), "Albumentations-based augmentation, and PyTorch training and evaluation with timm backbones.")
    
    c.drawString(34.00, py_to_c(480.10), "- Implemented and benchmarked 7 modern architectures from research literature for growth-stage classification, with the Swin")
    c.drawString(44.70, py_to_c(491.70), "Transformer achieving the highest accuracy of ~98%.")
    
    c.drawString(34.00, py_to_c(504.10), "- Used Grad-CAM to visualise and interpret model predictions, and built a Streamlit app for interactive inference.")
    
    # TECHNICAL SKILLS (User: "nothing on skills", keep regular)
    draw_heading("TECHNICAL SKILLS", 519.80)
    draw_line(531.60)
    
    c.setFont(FONT_REG, 9.5)
    c.drawString(X_LEFT, py_to_c(537.00), "Languages: Python, JavaScript, C++, SQL")
    c.drawString(X_LEFT, py_to_c(550.00), "Libraries: Scikit-learn, Pandas, NumPy, TensorFlow/Keras, OpenCV, MediaPipe, FastAPI")
    c.drawString(X_LEFT, py_to_c(563.00), "Tools: Git, VS Code, n8n, Docker, Hugging Face")
    c.drawString(X_LEFT, py_to_c(576.00), "Databases: MySQL")
    c.drawString(X_LEFT, py_to_c(589.00), "Domain: Artificial Intelligence, Machine Learning, Deep Learning, DSA")
    
    # CERTIFICATIONS & ACHIEVEMENTS (User: "nothing on certification", keep regular)
    draw_heading("CERTIFICATIONS & ACHIEVEMENTS", 605.10)
    draw_line(616.90)
    
    c.setFont(FONT_REG, 9.4)
    c.drawString(34.00, py_to_c(621.70), "- Dean's List — Academic Excellence, Bennett University | Sem I SGPA: 9.73")
    c.drawString(34.00, py_to_c(634.20), "- micro1 AI Interview Certification — Certified for Outstanding Performance in AI Interview, micro1 (2026)")
    c.drawString(34.00, py_to_c(646.70), "- McKinsey Forward Program Certification (2025)")
    c.drawString(34.00, py_to_c(659.20), "- Summer Analytics: Data Science & ML — IIT Guwahati (2025)")
    c.drawString(34.00, py_to_c(671.70), "- AWS Certified AI Practitioner Certificate")
    
    # POSITION OF RESPONSIBILITY (Bold on headings as requested)
    draw_heading("POSITION OF RESPONSIBILITY", 687.90)
    draw_line(699.70)
    
    # Council
    c.setFont(FONT_REG, 9.4)
    c.drawString(34.00, py_to_c(704.50), "-")
    c.setFont(FONT_BOLD, 9.4)
    c.drawString(42.00, py_to_c(704.50), "Bennett University Student Council — Sub-Head, IT")
    bw1 = c.stringWidth("Bennett University Student Council — Sub-Head, IT", FONT_BOLD, 9.4)
    c.setFont(FONT_REG, 9.4)
    c.drawString(42.00 + bw1, py_to_c(704.50), " (2025 – 2026) — Led IT operations for university events, coordinating")
    c.drawString(44.70, py_to_c(716.10), "cross-functional teams supporting 1,000+ participants across 5+ events. Contributed to multiple events and received appreciation")
    c.drawString(44.70, py_to_c(727.70), "for outstanding IT support and technical coordination.")
    
    # Campus Ambassador (user requested: improve line, remove 'the festival')
    c.drawString(34.00, py_to_c(740.10), "-")
    c.setFont(FONT_BOLD, 9.4)
    c.drawString(42.00, py_to_c(740.10), "Campus Ambassador — Techfest IIT Bombay:")
    bw2 = c.stringWidth("Campus Ambassador — Techfest IIT Bombay:", FONT_BOLD, 9.4)
    c.setFont(FONT_REG, 9.4)
    c.drawString(42.00 + bw2 + 4, py_to_c(740.10), "Represented Techfest, IIT Bombay on campus, promoted its events, and encouraged")
    c.drawString(44.70, py_to_c(751.70), "student participation.")
    
    c.save()
    print("Exact resume generated.")

out_pdf = "Lakshit_Malhotra_Resume_Polished.pdf"
draw_pdf(out_pdf)

downloads_path = r"C:\Users\laksh\Downloads\Lakshit_Malhotra_Resume_Polished.pdf"
shutil.copy(out_pdf, downloads_path)
print(f"Copied to {downloads_path}")

doc = fitz.open(out_pdf)
print("Total pages:", len(doc))
pix = doc[0].get_pixmap(dpi=200)
pix.save("resume_preview_exact.png")
print("Saved preview to resume_preview_exact.png")
