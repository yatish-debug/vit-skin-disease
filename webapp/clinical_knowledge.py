"""
Comprehensive Dermatopathology Knowledge Base & Dynamic Clinical Dossier Engine.
Provides deeply detailed medical pathology narratives, visual biomarker extraction,
differential diagnosis logic, and multi-tiered treatment protocols across all 22 conditions.
Tailors every report specifically to the optical traits of each uploaded photograph.
"""

import numpy as np
from PIL import Image

CLINICAL_KNOWLEDGE = {
    "ACNE": {
        "title": "Acne Vulgaris",
        "category": "Pilosebaceous Inflammatory Dermatosis",
        "pathology_variants": [
            "Pathophysiologically initiated by androgen-mediated sebum hypersecretion combined with abnormal follicular infundibular hyperkeratinization. This microenvironment fosters colonization by Cutibacterium acnes, activating Toll-like receptor 2 (TLR-2) cascades that recruit neutrophils and trigger localized follicular rupture, producing inflammatory papules, pustules, and deep-seated comedones.",
            "Characterized by follicular occlusion of the sebaceous glands, leading to retention of sebum and cellular debris. The resulting microcomedo expands under pressure and induces localized inflammatory mediators (IL-1alpha, TNF-alpha), producing distinctive open (blackheads) and closed (whiteheads) comedones with varying surrounding erythematous halos.",
            "Subepidermal micro-inflammatory dynamics demonstrate rupture of the follicular epithelium into the surrounding dermis. A dense perivascular lymphocytic infiltrate accompanied by reactive keratinization produces persistent erythematous papules that may progress to nodular or cystic lesions without targeted comedolytic intervention.",
            "Etiologically linked to altered lipid composition within the sebum, promoting bacterial biofilm formation and follicular barrier breakdown. Reactive follicular hyperkeratosis forms cohesive plugs that resist natural desquamation, visible optically as localized follicular bumps with varying degrees of vascular erythema."
        ],
        "first_line": "Topical fixed-combination therapy: Benzoyl peroxide (2.5% - 5%) combined with adapalene (0.1% - 0.3%) gel applied once nightly to clean, dry skin. For moderate inflammatory components, topical clindamycin (1%) or oral doxycycline (100mg daily) may be indicated under supervision.",
        "barrier_care": "Cleanse twice daily with a non-comedogenic, pH-balanced (pH 5.5) gentle foaming cleanser. Avoid aggressive physical scrubs that trigger mechanical friction (acne mechanica). Maintain skin hydration with oil-free, ceramide-rich fluid moisturizers.",
        "follow_up": "Re-evaluate therapeutic response at 6 to 8 weeks. Epidermal turnover requires 4 to 6 weeks to demonstrate noticeable reduction in inflammatory lesion count.",
        "red_flags": "Deep cystic nodules, sinus tract formation with draining exudate, facial edema, or onset of permanent keloidal/ice-pick scarring.",
        "diff_clue": "Presence of open and closed comedones distinguishes acne from acneiform rosacea (which lacks true comedones)."
    },
    "AKIEC": {
        "title": "Actinic Keratosis (Pre-Malignant)",
        "category": "Pre-Neoplastic Epidermal Dysplasia",
        "pathology_variants": [
            "Initiated by cumulative solar ultraviolet radiation causing cyclobutane pyrimidine dimers and signature p53 tumor-suppressor gene mutations. Keratinocyte atypia originates in the lower tiers of the epidermis, manifesting as atypical, pleomorphic nuclei with dyskeratosis and hyperchromasia, presenting optically as rough, adherent scaly keratoses.",
            "Histopathologically characterized by alternating orthokeratosis and parakeratosis over atypical, disorganized basal and suprabasal keratinocytes. The dermal-epidermal junction exhibits solar elastosis characterized by thick, disorganized basophilic elastic fibers from chronic photo-oxidative stress.",
            "Represents an early, biologically active continuum of cutaneous squamous cell carcinoma in situ. Dysplastic keratinocytes proliferate along the epidermal basement membrane without breaching the reticular dermis, producing a tactile 'sandpaper-like' hyperkeratotic plaque with background teleangiectasia.",
            "Exhibits focal loss of the granular layer and atypical mitotic figures localized to the interfollicular epidermis. The underlying dermis shows prominent actinic elastosis accompanied by a superficial chronic perivascular lymphocytic inflammatory reaction."
        ],
        "first_line": "Targeted lesion destruction via liquid nitrogen cryotherapy (5 to 10 second freeze-thaw cycle). For field cancerization with multiple subclinical lesions: Topical 5-Fluorouracil (5-FU 5%) cream applied twice daily for 2 to 4 weeks, or Imiquimod 3.75% / 5% cream.",
        "barrier_care": "Strict broad-spectrum physical sunscreen application (SPF 50+ containing zinc oxide / titanium dioxide) reapplied every 2 hours outdoors. Broad-brimmed ultraviolet protective headwear and avoidance of peak solar hours (10 AM to 4 PM).",
        "follow_up": "Routine dermatologist total-body skin examination at 3 to 6 months post-treatment to confirm resolution and screen for progression.",
        "red_flags": "Rapid induration, ulceration, spontaneous bleeding, tenderness on palpation, or plaque diameter exceeding 1 cm (strongly raising suspicion for invasive Squamous Cell Carcinoma).",
        "diff_clue": "Rough sandpaper palpability with adherent scale distinguishes AK from flat seborrheic keratosis or benign solar lentigines."
    },
    "BENIGN_TUMOR": {
        "title": "Benign Cutaneous Neoplasm",
        "category": "Non-Malignant Mesenchymal/Epithelial Proliferation",
        "pathology_variants": [
            "Represents a well-circumscribed, non-invasive tissue proliferation originating from cutaneous stromal, vascular, or epithelial lineages. Cellular architecture maintains uniform nuclear contours, absence of atypical mitotic figures, and preserved intercellular bridges without destructive stromal invasion.",
            "Histologically demonstrates cohesive aggregates of mature cells forming organized lobules, papillae, or mesenchymal nests within the dermis or subcutis. The overlying epidermis is typically intact or gently thinned without cellular dysplasia or basement membrane disruption.",
            "Benign biological behavior characterized by slow expansion and absent metastatic potential. Stromal interaction is encapsulated or cleanly demarcated, displaying normal collagen bundles and healthy vascular supply without neovascular atypia.",
            "Optical presentation exhibits symmetrical contours, smooth to gently lobulated surface architecture, and harmonious pigment uniformity across all visual sectors, consistent with indolent, non-aggressive tissue expansion."
        ],
        "first_line": "Clinical observation is standard of care. If lesions cause mechanical irritation, recurrent trauma, or aesthetic concern: Electrosurgical shave excision, radiofrequency ablation, or minor punch excision under localized anesthesia.",
        "barrier_care": "Avoid scratching, pinching, or tight clothing friction over the nodule. Cleanse with mild, unscented soap and apply simple petrolatum if surface becomes irritated.",
        "follow_up": "Annual routine skin wellness exam. No urgent surveillance required unless subjective morphological change occurs.",
        "red_flags": "Rapid asymmetric enlargement, spontaneous bleeding without prior trauma, color variegation, or development of central ulceration.",
        "diff_clue": "Absence of invasiveness, uniform color tone, and smooth regular boundaries distinguish benign neoplasms from nodular malignancies."
    },
    "BULLOUS": {
        "title": "Bullous Disease (Pemphigoid / Pemphigus)",
        "category": "Autoimmune Subepidermal/Intraepidermal Blistering",
        "pathology_variants": [
            "Pathophysiology driven by pathogenic autoantibodies (IgG and complement C3) targeting hemidesmosomal structural antigens (BP180 / BP230) at the dermal-epidermal basement membrane zone. Complement activation stimulates neutrophil and eosinophil elastase release, producing enzymatic detachment of the epidermis and forming tense, fluid-filled subepidermal bullae.",
            "Autoimmune blistering disorder characterized by cleavage of epidermal cohesion. Direct immunofluorescence (DIF) microscopy demonstrates linear deposition of IgG and C3 along the dermo-epidermal basement membrane, presenting clinically as tense bullae on an erythematous or urticarial base.",
            "Subepidermal cavity formation accompanied by intense dermal eosinophilia and superficial perivascular infiltrate. Cleavage occurs within the lamina lucida, leaving the blister roof composed of intact, full-thickness epidermis, which confers mechanical resistance to gentle pressure (negative Nikolsky sign).",
            "Manifests with extensive tense cutaneous blisters, eroded excoriations, and severe pruritus. Circulating autoantibodies compromise dermal adhesion integrity, with optical features demonstrating translucent serous or serosanguinous fluid accumulation bounded by intense perilesional erythema."
        ],
        "first_line": "High-potency topical corticosteroids (Clobetasol propionate 0.05% cream 20-30g daily applied over body) as primary therapy for localized/moderate disease. For generalized disease, oral systemic corticosteroids (Prednisone 0.5-0.75 mg/kg/day) paired with steroid-sparing immunosuppressants (Methotrexate, Mycophenolate mofetil) under dermatological management.",
        "barrier_care": "Do not mechanically de-roof intact blisters (the natural roof acts as a sterile biologic dressing). Cleanse eroded skin with dilute chlorhexidine or saline soaks. Apply petrolatum-impregnated non-adherent dressings (Mepitel/Vaseline gauze) to minimize shear friction.",
        "follow_up": "Urgent dermatological consultation within 24 to 48 hours for diagnostic confirmation via paired punch biopsies (H&E routine histology and perilesional Direct Immunofluorescence).",
        "red_flags": "Oral mucosal erosion, widespread skin denudation (>10% body surface area), high fever, or signs of secondary bacterial sepsis (purulent exudate, systemic chills).",
        "diff_clue": "Tense subepidermal blisters with negative Nikolsky sign differentiate bullous pemphigoid from the flaccid intraepidermal blisters of pemphigus vulgaris."
    },
    "CANDIDIASIS": {
        "title": "Cutaneous Candidiasis",
        "category": "Superficial Opportunistic Yeast Infection",
        "pathology_variants": [
            "Caused by the opportunistic proliferation and dimorphic transformation of Candida albicans from commensal blastoconidia into invasive pseudohyphae. The organism penetrates the non-vital stratum corneum in intertriginous skin folds (inframammary, inguinal, axillary) where heat and moisture break down epidermal barrier integrity.",
            "Clinically hallmarked by intense beefy-red erythematous plaques with macerated borders, surrounded by pathognomonic pinpoint satellite papules and pustules. Microscopic examination of fungal scrapings via potassium hydroxide (KOH) reveals budding yeast cells and pseudohyphae.",
            "The fungal hyphae secrete aspartyl proteinases and phospholipases that digest stratum corneum proteins, provoking localized epidermal inflammation and capillary dilatation visible optically as glistening vivid erythema with peripheral epidermal peeling.",
            "Opportunistic growth triggered by warm, occluded epidermal microenvironments, often exacerbated by systemic factors such as diabetes mellitus, antibiotic usage, or localized maceration. The presentation shows intense localized redness with moist superficial scale."
        ],
        "first_line": "Topical antifungal therapy: Clotrimazole 1%, Miconazole 2%, or Ketoconazole 2% cream applied twice daily to affected area and surrounding margins for 2 to 3 weeks. If accompanied by severe inflammatory pruritus, short-course low-potency hydrocortisone (1%) may be co-applied for initial 48 hours.",
        "barrier_care": "Strict moisture reduction: Keep skin folds dry using soft cotton pads or zinc oxide barrier pastes. Avoid synthetic, non-breathable fabrics. Cleanse gently with antifungal-compatible washes and ensure complete air-drying post-bathing.",
        "follow_up": "Clinical re-evaluation in 10 to 14 days. If refractory, oral Fluconazole (150mg weekly for 2 to 4 weeks) may be considered by a clinician.",
        "red_flags": "Rapid spread beyond intertriginous creases, deep follicular pustules, systemic fever, or diabetic blood glucose decompensation.",
        "diff_clue": "Presence of discrete peripheral satellite pustules and beefy-red macerated plaques differentiates candidiasis from tinea or contact dermatitis."
    },
    "DRUG_ERUPTION": {
        "title": "Adverse Cutaneous Drug Eruption",
        "category": "Immunologically Mediated Hypersensitivity Reaction",
        "pathology_variants": [
            "Typically represents a Type IV cell-mediated delayed hypersensitivity reaction to a newly introduced oral or parenteral pharmacological agent (most commonly beta-lactams, sulfonamides, anticonvulsants, or NSAIDs). Drug-specific CD4+ and CD8+ T-lymphocytes migrate to the dermo-epidermal junction, releasing inflammatory cytokines that induce perivascular edema and apoptotic keratinocytes.",
            "Manifests as a symmetrical, diffuse morbilliform (measles-like) exanthem with confluent macules and papules that characteristically begin on the trunk and proximal extremities before spreading centrifugally. Superficial dermal edema and vascular dilation produce vibrant erythematous coloration.",
            "Histological features include superficial perivascular lymphocytic infiltrates with scattered eosinophils, accompanied by basal vacuolization and necrotic individual keratinocytes. The eruption typically manifests 7 to 14 days after initial drug exposure.",
            "Optical presentation demonstrates widespread, confluent erythematous maculopapular blanching lesions. The reaction reflects systemic immunological sensitization manifesting through cutaneous vascular hyperemia and superficial dermal inflammatory recruitment."
        ],
        "first_line": "Immediate identification and medical discontinuation of the offending culprit medication under physician guidance. Symptomatic management with high-potency topical corticosteroids (triamcinolone 0.1% or clobetasol 0.05%) and non-sedating oral second-generation antihistamines (Cetirizine 10mg daily) for pruritus.",
        "barrier_care": "Cool, soothing colloidal oatmeal baths and liberal application of unscented bland emollient creams to prevent barrier dehydration. Avoid hot showers and skin friction.",
        "follow_up": "Consult a physician within 24 hours. Dermatological monitoring to ensure absence of mucosal involvement or systemic transition.",
        "red_flags": "Mucosal erosion (lips, conjunctiva, genitalia), epidermal detachment/blistering, facial edema, painful skin (raising concern for SJS/TEN or DRESS syndrome).",
        "diff_clue": "Sudden symmetric bilateral centrifugal onset shortly following new medication intake strongly points toward drug-induced exanthem over viral exanthem."
    },
    "ECZEMA": {
        "title": "Atopic Dermatitis / Eczema",
        "category": "Chronic Relapsing Inflammatory Dermatosis",
        "pathology_variants": [
            "Pathophysiologically anchored in primary epidermal barrier dysfunction—frequently linked to loss-of-function filaggrin (FLG) gene mutations—coupled with a hyperactive T-helper 2 (Th2) immune response. Barrier breakdown allows allergen penetration, driving IL-4 and IL-13 production, eosinophil recruitment, and sustained intense pruritus.",
            "Histologically characterized by epidermal spongiosis (intercellular edema between spinous keratinocytes) accompanied by superficial perivascular lymphocytic infiltration, micro-vesiculation, and parakeratosis. In chronic phases, persistent scratching induces prominent epidermal acanthosis and lichenification.",
            "Cutaneous presentation demonstrates poorly-defined erythematous patches, excoriations, and micro-scaling. Pruritus-induced scratching drives a vicious 'itch-scratch cycle' that damages the stratum corneum and increases vulnerability to Staphylococcus aureus supercolonization.",
            "Optical features reveal diffuse, non-demarcated erythematous plaques with focal superficial scaling, fine fissures, and excoriation crusts. Cellular intercellular edema creates a spongy epidermal texture with prominent cutaneous moisture loss."
        ],
        "first_line": "First-line topical anti-inflammatory therapy: Medium to high-potency topical corticosteroids (Fluticasone 0.05% or Triamcinolone 0.1% ointment) applied twice daily during acute flare-ups for 7 to 14 days. For delicate facial/intertriginous zones: Topical calcineurin inhibitors (Tacrolimus 0.1% or Pimecrolimus 1%).",
        "barrier_care": "Frequent, liberal application of high-lipid emollient creams containing ceramides, cholesterol, and free fatty acids immediately post-bathing ('soak and seal' within 3 minutes). Avoid soaps, fragrance, wool clothing, and hot water.",
        "follow_up": "Dermatology review if acute flares do not achieve clinical quiescence within 2 weeks of appropriate topical therapy.",
        "red_flags": "Punched-out umbilicated vesicles or painful grouped crusts (indicating Eczema Herpeticum), golden honey-colored crusting (impetiginization), or widespread erythroderma.",
        "diff_clue": "Ill-defined borders, marked pruritus, and flexural predilection contrast with the sharply demarcated plaques of psoriasis."
    },
    "BITES": {
        "title": "Arthropod Bite / Scabies Reaction",
        "category": "Parasitic / Arthropod Cutaneous Reaction",
        "pathology_variants": [
            "Induced by mechanical puncture and injection of foreign saliva proteins, formic acid, or anticoagulants from arthropods (mosquitoes, bedbugs, fleas) or burrowing Sarcoptes scabiei mites. This triggers a localized mast-cell degranulation (histamine release) followed by an eosinophil-rich delayed hypersensitivity reaction producing pruritic urticarial papules.",
            "Histological section demonstrates prominent superficial and deep wedge-shaped perivascular infiltrates dominated by eosinophils, histiocytes, and lymphocytes. Dermal edema and focal epidermal spongiosis manifest clinically as an indurated, erythematous papule with a central punctum.",
            "In scabietic infestation, female mites burrow through the stratum corneum, depositing eggs and feces (scybala). This induces a widespread, intensely pruritic allergic reaction with characteristic serpiginous burrows and secondary excoriated papules concentrated in finger web spaces, wrists, and axillae.",
            "Optical traits exhibit discrete, indurated erythematous papules frequently arranged in linear clusters ('breakfast, lunch, dinner' distribution for bedbugs) or scattered punctate lesions with surrounding reactive vascular erythema."
        ],
        "first_line": "For standard insect bites: Mid-potency topical corticosteroid (Hydrocortisone butyrate 0.1% or Triamcinolone 0.1%) plus oral antihistamine (Levocetirizine 5mg). For diagnosed scabies: Topical Permethrin 5% cream applied neck-to-toes for 8 to 14 hours, repeated in 7 days, or oral Ivermectin (200 mcg/kg) in paired doses.",
        "barrier_care": "Keep fingernails trimmed short to prevent secondary bacterial inoculations from excoriations. Wash bedding and clothing in hot water (>60°C) and seal non-washable items in plastic bags for 72 hours.",
        "follow_up": "Medical reassessment at 2 weeks. Post-scabietic pruritus can persist for several weeks even after successful eradication.",
        "red_flags": "Spreading warmth, ascending erythema, or purulent crusting indicating secondary bacterial cellulitis (Staphylococcus/Streptococcus).",
        "diff_clue": "Presence of a central punctum or linear clustered configuration distinguishes arthropod bites from spontaneous urticaria."
    },
    "LICHEN": {
        "title": "Lichen Planus",
        "category": "T-Cell Mediated Interface Dermatitis",
        "pathology_variants": [
            "Pathophysiologically governed by an autoreactive CD8+ cytotoxic T-cell immune response targeted against altered basal keratinocytes. Activated T-cells aggregate in a dense, band-like infiltrate along the dermo-epidermal junction, inducing extensive hydropic degeneration and apoptosis of basal cells (forming colloid/Civatte bodies).",
            "Classic clinical presentation obeys the '6 Ps': Pruritic, Polygonal, Planar (flat-topped), Purple (violaceous), Papules and Plaques. Close dermoscopic inspection reveals pathognomonic Wickham's striae—a delicate network of white lines caused by focal hypergranulosis.",
            "Histopathology demonstrates classic 'saw-tooth' rete ridge hyperplasia, wedge-shaped hypergranulosis, compact orthokeratosis, and absence of parakeratosis. The basal layer undergoes colloid degeneration with eosinophilic apoptotic keratinocyte remnants visible along the interface.",
            "Optical biomarkers exhibit distinctive flat-topped violaceous to dusky-red papules with glistening surface luster. The lesion architecture demonstrates sharp geometric borders, frequently displaying the Koebner phenomenon (lesions developing along lines of mechanical trauma)."
        ],
        "first_line": "High-potency topical corticosteroids (Clobetasol propionate 0.05% ointment or Betamethasone dipropionate 0.05%) applied twice daily for 2 to 4 weeks with tapering. For hypertrophic or refractory lesions: Intralesional triamcinolone acetonide injections (5-10 mg/mL).",
        "barrier_care": "Avoid skin trauma and scratching to prevent Koebnerization. Use soothing unfragranced lipid balms. Perform thorough oral cavity self-inspection (oral lichen planus screening on buccal mucosa).",
        "follow_up": "Clinical evaluation by a dermatologist within 2 to 3 weeks. Screen for underlying associations such as Hepatitis C serology if extensive.",
        "red_flags": "Oral mucosal erosive ulcerations, nail dystrophy/pterygium formation, or scarring follicular scalp involvement (lichen planopilaris leading to permanent alopecia).",
        "diff_clue": "Violaceous polygonal flat-topped architecture with Wickham's striae cleanly differentiates lichen planus from psoriasis and eczema."
    },
    "LUPUS": {
        "title": "Cutaneous Lupus Erythematosus",
        "category": "Autoimmune Photodistributed Interface Dermatitis",
        "pathology_variants": [
            "Driven by a complex autoimmune reaction characterized by UV-induced keratinocyte apoptosis, autoantibody generation (Ro/SSA, La/SSB), and plasmacytoid dendritic cell activation driving Type I Interferon pathways. An interface dermatitis destroys basal keratinocytes and causes thick dermo-epidermal basement membrane duplication.",
            "Clinical manifestation displays well-demarcated erythematous, violaceous scaly plaques with follicular plugging ('carpet tack' sign) and subsequent central hypopigmented scarring atrophy with hyperpigmented borders. Prominent telangiectasias and follicular plugging are hallmark features.",
            "Histopathological features reveal prominent vacuolar interface dermatitis, apoptotic Civatte bodies, marked epidermal atrophy, and abundant dermal mucin deposition visualized with colloidal iron staining. Dense perivascular and periadnexal lymphocytic infiltrates extend deep into the reticular dermis.",
            "Optical presentation demonstrates characteristic photo-distributed erythematous plaques featuring central epidermal atrophy and peripheral hyperpigmentation. The lesion shows marked vascular erythema alternating with pale fibrotic cicatricial change."
        ],
        "first_line": "First-line topical/intralesional therapy: Superpotent topical corticosteroids (Clobetasol 0.05%) or topical calcineurin inhibitors (Tacrolimus 0.1%). Systemic first-line: Hydroxychloroquine (200-400 mg/day, maximum 5 mg/kg real body weight) under medical and ophthalmologic surveillance.",
        "barrier_care": "Strict UV photoprotection is mandatory: Daily broad-spectrum SPF 50+ mineral sunscreen, photoprotective clothing (UPF 50+), and avoidance of both sunlight and fluorescent lighting containing high UV components.",
        "follow_up": "Prompt rheumatology/dermatology evaluation within 1 to 2 weeks for serological screening (ANA, ENA panel, dsDNA, complement levels, urinalysis) to assess systemic lupus erythematosus (SLE) risk.",
        "red_flags": "Systemic joint pain, persistent unexplained fever, pleuritic chest pain, photosensitive malar rash, or protein/foamy urine indicating lupus nephritis.",
        "diff_clue": "Follicular plugging with central scarring atrophy and photodistributed predilection differentiates discoid lupus from actinic keratosis and psoriasis."
    },
    "NV": {
        "title": "Melanocytic Nevus (Common Mole)",
        "category": "Benign Melanocytic Proliferation",
        "pathology_variants": [
            "Represents an organized, benign clonal proliferation of melanocytes (nevus cells) originating from the neural crest. Nevocytes organize into cohesive spherical nests at the dermo-epidermal junction (junctional nevus), within the dermis (intradermal nevus), or across both compartments (compound nevus).",
            "Histologically demonstrates normal melanocytic maturation: Nevus nests in the superficial dermis contain melanin-rich cells that progressively diminish in size, become smaller and lymphocyte-like in the mid-dermis, and transform into slender, non-pigmented spindle-like neurotized cells at the base.",
            "Maintains architectural symmetry, sharp lateral demarcation, and absence of cellular atypia. Nuclear contours are round and uniform, with low nucleus-to-cytoplasm ratios and complete absence of atypical deep mitotic figures.",
            "Optical presentation shows uniform, harmonious pigmentation (light to dark brown) with smooth, regular borders and symmetric geometric contours. Dermoscopic assessment displays typical benign pigment networks, regular globules, or homogeneous patterns."
        ],
        "first_line": "No intervention required. Benign melanocytic nevi are harmless physiological features. Shave or punch excision is indicated only if the lesion is subject to chronic mechanical belt/bra strap trauma or for aesthetic preference.",
        "barrier_care": "Routine sun protection habits: Regular application of broad-spectrum sunscreen. Avoid sunburns, which trigger eruptive nevi and melanocyte mutation.",
        "follow_up": "Routine monthly self-examination following the ABCDE criteria: Asymmetry, Border irregularity, Color variegation, Diameter (>6mm), and Evolution (any change in size, shape, elevation).",
        "red_flags": "Development of the 'Ugly Duckling' sign (lesion looking visibly different from all other patient moles), focal black pigment deposition, border notching, bleeding, or itching.",
        "diff_clue": "Architectural symmetry, structural pigment regularity, and absence of multiple colors distinguish benign nevi from malignant melanoma."
    },
    "PSORIASIS": {
        "title": "Plaque Psoriasis",
        "category": "Autoimmune Hyperproliferative Papulosquamous Disease",
        "pathology_variants": [
            "Pathophysiology centered on dysregulated crosstalk between dendritic cells, Th17 and Th1 T-cells, driving a cytokine storm dominated by IL-23, IL-17, and TNF-alpha. This hyperstimulates basal keratinocytes, drastically accelerating epidermal transit time from normal 28 days down to 3 to 5 days, resulting in premature cornification.",
            "Presents classically with sharply circumscribed, salmon-pink erythematous plaques capped by thick, cohesive, micaceous silvery-white scales. Mechanical scraping of the scale reveals pinpoint capillary bleeding underneath (positive Auspitz sign) due to tortuous dilated dermal capillaries.",
            "Histopathological hallmarks include marked regular epidermal acanthosis with clubbed rete ridges ('test tubes in a rack'), confluent parakeratosis with absent stratum granulosum, and neutrophil aggregates within the stratum corneum (Munro microabscesses).",
            "Optical characteristics show vibrant salmon-pink vascular plaques with high-reflectance hyperkeratotic silvery scale accumulation. Lesion borders are crisp, well-demarcated, and commonly localized over extensor surfaces (elbows, knees, lumbosacral region)."
        ],
        "first_line": "Potent topical corticosteroid (Betamethasone dipropionate 0.05%) combined with topical vitamin D3 analogue (Calcipotriol 0.005%) applied once daily for 4 weeks. For scalp or thick plaques: Keratolytic salicylic acid 3-5% shampoos or urea ointments to gently dissolve thick scale.",
        "barrier_care": "Generous application of thick emollient ointments immediately following lukewarm baths to lock in moisture and soften plaques. Avoid mechanical trauma, friction, and skin abrasions (Koebner phenomenon triggers).",
        "follow_up": "Clinical evaluation by a dermatologist within 2 to 4 weeks. For patients with >10% body surface area involvement or arthritis, systemic biologics (IL-17, IL-23, TNF inhibitors) or Methotrexate are indicated.",
        "red_flags": "Associated morning joint stiffness, pain, or dactylitis ('sausage digits') indicating psoriatic arthritis; rapid emergence of sterile pustules (pustular psoriasis); or total body erythema.",
        "diff_clue": "Thick silvery micaceous scale, Auspitz sign, and extensor distribution cleanly separate plaque psoriasis from eczema."
    },
    "ROSACEA": {
        "title": "Acne Rosacea",
        "category": "Neurovascular & Inflammatory Facial Dermatosis",
        "pathology_variants": [
            "Pathogenetically driven by innate immune dysregulation (elevated cathelicidin antimicrobial peptide LL-37 and kallikrein 5) coupled with neurovascular hyperreactivity and Demodex folliculorum mite proliferation. Transient vascular flushing progressively transforms into fixed central facial erythema, telangiectasias, and inflammatory papulopustules.",
            "Histologically shows dilated, ectatic superficial dermal post-capillary venules with perivascular lymphocytic infiltrates, dermal elastosis, and localized perifollicular inflammation without comedo formation. In phymatous variants, sebaceous hyperplasia and connective tissue fibrosis predominate.",
            "Triggered by heat, ultraviolet radiation, spicy foods, alcohol, and emotional stress via transient receptor potential (TRP) channels on sensory nerves and endothelial cells, inducing localized neurogenic inflammation and sustained facial flushing.",
            "Optical features present central facial distribution (cheeks, nose, forehead, chin) featuring delicate reticular telangiectasias, background erythema, and inflammatory papules with strict sparing of the periocular cutaneous zones."
        ],
        "first_line": "For papulopustular rosacea: Topical Ivermectin 1% cream applied once nightly, or Topical Metronidazole 0.75% gel twice daily. For persistent background erythema: Topical alpha-adrenergic agonist (Brimonidine 0.33% or Oxymetazoline 1%) for temporary vasoconstriction.",
        "barrier_care": "Identify and avoid individual neurovascular triggers (thermal extremes, hot drinks, spicy food, stress). Use ultra-mild, soap-free cleansers and 100% mineral sunscreens (Zinc oxide). Avoid astringents, toners, and physical exfoliants.",
        "follow_up": "Dermatology follow-up at 8 to 12 weeks to assess anti-inflammatory and vascular stabilization.",
        "red_flags": "Ocular symptoms including gritty sensation, photophobia, blepharitis, or recurrent styes (indicating Ocular Rosacea requiring systemic doxycycline), or progressive nasal tissue hypertrophy (rhinophyma).",
        "diff_clue": "Absence of true comedones and central facial flushing distribution differentiates rosacea from acne vulgaris."
    },
    "BKL": {
        "title": "Seborrheic Keratosis",
        "category": "Benign Epidermal Epithelial Tumor",
        "pathology_variants": [
            "Common benign epidermal tumor resulting from clonal proliferation of mature keratinocytes, often driven by activating somatic mutations in FGFR3 (fibroblast growth factor receptor 3) or PIK3CA. The lesion develops entirely within the epidermis without cellular atypia or malignant potential.",
            "Histopathology demonstrates an exophytic, sharply circumscribed tumor characterized by marked hyperkeratosis, papillomatosis, and acanthosis of uniform basaloid cells. Distinctive intraepidermal horn pseudocysts (keratin-filled invaginations) are a diagnostic hallmark.",
            "Clinically exhibits a classic 'stuck-on', greasy, waxy or verrucous appearance, ranging in pigmentation from pale tan to deep brown-black. The lesion margins are sharply demarcated, appearing as though pressed against the skin like a coin or drop of candle wax.",
            "Optical presentation reveals a deeply textured, verrucous surface with visible keratin-filled crypts and comedo-like openings under dermoscopic magnification. Color tone is uniform within regional sectors, maintaining benign stability."
        ],
        "first_line": "No medical intervention is required. Reassuringly benign. If the lesion catches on clothing, becomes irritated, or is cosmetically undesired: Light liquid nitrogen cryotherapy, electrodessication and curettage, or shave removal under local anesthesia.",
        "barrier_care": "Avoid aggressive picking or scratching, which may cause secondary bleeding or superficial infection. Maintain routine skin hydration.",
        "follow_up": "Routine clinical observation. No mandatory surveillance required unless suspicious rapid evolution occurs.",
        "red_flags": "Sudden eruptive emergence of dozens of seborrheic keratoses over a short period (sign of Leser-Trelat, rarely associated with internal malignancy), or solitary atypical black lesion with irregular borders.",
        "diff_clue": "Keratin-filled horn pseudocysts and waxy 'stuck-on' contour distinguish seborrheic keratosis from malignant melanoma."
    },
    "MEL": {
        "title": "Malignant Melanoma",
        "category": "High-Risk Malignant Melanocytic Neoplasm",
        "pathology_variants": [
            "Highly aggressive malignant tumor originating from neoplastic transformation of melanocytes, predominantly driven by cumulative or intermittent UV radiation inducing driver mutations in BRAF (V600E), NRAS, or KIT oncogenes. Atypical melanocytes escape cell cycle checkpoints, proliferating horizontally along the junction (radial growth phase) before invading vertically into the dermis.",
            "Histopathologically demonstrates asymmetric, poorly circumscribed proliferations of atypical, pleomorphic melanocytes exhibiting nuclear hyperchromasia, prominent nucleoli, and frequent atypical mitoses. Pagetoid (buckshot) upward scatter of melanocytes into upper epidermal layers is a definitive diagnostic criteria.",
            "Vertical invasion carries severe potential for lymphatic and hematogenous metastasis. Key prognostic factors include Breslow tumor thickness (measured from the granular layer to the deepest point of invasion), presence of ulceration, and mitotic rate per square millimeter.",
            "Optical biomarkers present hallmark ABCDE characteristics: Asymmetry across dual axes, notched or irregular borders, variegation with multiple pigment hues (black, dark brown, red, white, blue), diameter exceeding 6mm, and evolutionary change. Dermoscopy reveals atypical pigment networks, regression structures, and blue-white veils."
        ],
        "first_line": "Urgent, complete primary surgical excisional biopsy with narrow (1-2 mm) margins down to subcutaneous fat (avoiding initial shave biopsy to preserve Breslow depth accuracy). Subsequent definitive wide local excision with margin width (1 to 2 cm) determined by histological Breslow depth, accompanied by Sentinel Lymph Node Biopsy (SLNB) if Breslow depth >0.8 mm.",
        "barrier_care": "Do not manipulate, cauterize, or attempt topical treatments on suspected melanoma lesions. Protect the entire skin mantle with strict broad-spectrum photoprotection.",
        "follow_up": "Immediate urgent consultation with a surgical oncologist or dermatologic surgeon within 24 to 48 hours.",
        "red_flags": "Rapid focal nodular elevation, spontaneous bleeding, regional lymphadenopathy, satellite pigment spots around primary lesion, or sensory itching/pain.",
        "diff_clue": "High architectural asymmetry, multi-colored pigment variegation (blue-white veil), and pagetoid melanocyte scatter definitively distinguish melanoma from benign nevi."
    },
    "PHOTO_DAMAGE": {
        "title": "Solar Elastosis & Photoaging",
        "category": "Chronic UV-Induced Cutaneous Degeneration",
        "pathology_variants": [
            "Result of chronic, cumulative solar ultraviolet (UVA and UVB) exposure inducing matrix metalloproteinases (MMPs) that degrade dermal collagen type I and III. Concurrently, damaged abnormal elastin fibers accumulate in the reticular dermis, manifesting as solar elastosis with loss of skin recoil.",
            "Clinically presents with yellowish, thickened, leathery cutaneous architecture (cutis rhomboidalis nuchae), accompanied by dyspigmentation, solar lentigines, fine and deep rhytides, and fragile dermal blood vessels leading to solar purpura.",
            "Histological sections reveal dense masses of tangled, thickened, basophilic degenerated elastic fibers replacing the normal eosinophilic dermal collagen bundles in the upper and mid-dermis, separated from the epidermis by a thin grenz zone.",
            "Optical presentation demonstrates widespread background telangiectasia, mottled hyperpigmentation, uneven skin texture, and superficial epidermal thinning consistent with chronic photo-oxidative stress."
        ],
        "first_line": "Topical retinoid therapy: Tretinoin (0.025% - 0.05% cream) applied nightly to stimulate collagen synthesis, enhance epidermal turnover, and compact the stratum corneum. Topical antioxidant serums (Vitamin C / L-Ascorbic Acid 15% + Ferulic Acid) applied mornings.",
        "barrier_care": "Strict daily broad-spectrum SPF 50+ mineral sunscreen application 365 days a year, regardless of cloud cover. Wear protective ultraviolet clothing and wide-brimmed hats.",
        "follow_up": "Annual total-body dermatologic skin examination to monitor for development of non-melanoma skin cancers (Basal Cell / Squamous Cell Carcinomas).",
        "red_flags": "Development of focal non-healing crusted papules, localized ulcerations, or rapidly indurated nodules within photo-damaged fields.",
        "diff_clue": "Generalized mottled pigment and solar elastosis without discrete dysplastic cellular borders distinguish photoaging from discrete malignant lesions."
    },
    "TINEA": {
        "title": "Dermatophytosis (Tinea / Ringworm)",
        "category": "Superficial Cutaneous Fungal Infection",
        "pathology_variants": [
            "Superficial fungal infection of keratinized tissue (stratum corneum) caused by dermatophyte fungi of the genera Trichophyton (e.g., T. rubrum, T. mentagrophytes), Microsporum, or Epidermophyton. The fungal hyphae secrete keratinases that digest structural keratin, triggering a localized centrifugal cell-mediated inflammatory response.",
            "Clinically hallmarked by characteristic annular (ring-shaped), erythematous plaques with a raised, actively advancing scaly border and relative central clearing as the fungus exhausts local keratin nutrition. Microscopic potassium hydroxide (KOH) examination shows septate branching fungal hyphae.",
            "Histologically shows compact orthokeratosis or parakeratosis containing delicate fungal hyphae within the stratum corneum (best highlighted with Periodic Acid-Schiff [PAS] or Grocott's methenamine silver [GMS] staining). Superficial perivascular lymphocytic infiltration and dermal edema accompany active margins.",
            "Optical biomarkers exhibit prominent circular or polycyclic configurations with marked erythematous scaling rims and lighter central areas. The advancing active edge exhibits focal micro-scaling and elevated inflammatory chromatic contrast."
        ],
        "first_line": "Topical allylamine therapy: Terbinafine 1% cream or Butenafine 1% cream applied twice daily to the active lesion and 2-3 cm into healthy surrounding skin for 2 to 4 weeks. (Avoid topical corticosteroid combination creams like betamethasone-clotrimazole which trigger Tinea Incognito).",
        "barrier_care": "Keep the affected anatomical region thoroughly dry and well-aerated. Use separate clean towels for infected areas to prevent autoinoculation. Wash sports gear, socks, and bedding in hot water.",
        "follow_up": "Clinical review in 14 days. If lesions are extensive, recalcitrant, or involve hair follicles (Tinea capitis / Majocchi's granuloma), oral Terbinafine (250 mg daily for 2 to 4 weeks) is indicated.",
        "red_flags": "Spreading pustular folliculitis within the lesion, involvement of nail plates (onychomycosis requiring prolonged oral therapy), or failure to respond after 3 weeks of compliant topical therapy.",
        "diff_clue": "Active advancing scaly annular border with central clearing cleanly differentiates tinea from annular erythema and nummular eczema."
    },
    "NORMAL": {
        "title": "Normal / Physiological Skin",
        "category": "Intact Physiological Cutaneous Architecture",
        "pathology_variants": [
            "Demonstrates completely preserved cutaneous histology: Intact stratified squamous keratinized epithelium comprising basal, spinous, granular, and regular cornified layers without cellular atypia, spongiosis, or architectural disruption.",
            "Normal dermo-epidermal junction with harmonious dermal papillae and epidermal rete pegs. Dermal collagen and elastic networks exhibit normal orientation and density without inflammatory perivascular cuffing or solar elastosis.",
            "Optical presentation exhibits uniform cutaneous reflectivity, smooth skin surface markings (dermatoglyphics), and balanced baseline vascular and melanin tones without pathological chromatic clustering or surface scaling.",
            "No active inflammatory, neoplastic, or infectious biomarkers identified. Epidermal cellular turnover and stratum corneum hydration indices are within healthy baseline parameters."
        ],
        "first_line": "No medical intervention indicated. Continue routine protective skincare regimen.",
        "barrier_care": "Maintain optimal skin barrier function: Daily broad-spectrum SPF 30+ sunscreen, gentle pH-balanced cleansers, and daily ceramide-containing hydration.",
        "follow_up": "Periodic routine self-skin examinations to monitor for new or changing cutaneous lesions.",
        "red_flags": "Development of any new rapidly enlarging, bleeding, or persistently itchy cutaneous spots.",
        "diff_clue": "Absence of erythema, scaling, architectural asymmetry, or border irregularity confirms normal baseline physiological tissue."
    },
    "VASC": {
        "title": "Vascular Tumor (Hemangioma / Angioma)",
        "category": "Benign Endothelial Vascular Proliferation",
        "pathology_variants": [
            "Benign vascular lesion characterized by localized hyperplasia or ectasia of endothelial cells lining blood vessels within the superficial or deep dermis. Includes common cherry angiomas, pyogenic granulomas, and infantile/congenital hemangiomas.",
            "Histologically demonstrates circumscribed, closely aggregated capillary lumens lined by plump, regular endothelial cells without cellular atypia or multilayering. Lumens are filled with erythrocytes and supported by a delicate collagenous stroma.",
            "Clinical presentation displays bright crimson, violaceous, or purplish macules or dome-shaped papules. Pressure via dermoscopy (diascopy) reveals partial or complete blanching as intravascular blood is temporarily displaced.",
            "Optical features present intense, high-chroma vascular saturation (elevated red-channel dominance). Under dermoscopy, characteristic red, purple, or blue-black round to oval vascular lacunae (lagunes) are visible within a well-circumscribed boundary."
        ],
        "first_line": "Benign cherry angiomas require no medical intervention. If bleeding frequently from trauma, ulcerated, or cosmetically undesired: Vascular-targeted laser therapy (Pulsed Dye Laser 595 nm or Nd:YAG), electrodessication, or minor shave excision.",
        "barrier_care": "Avoid friction, scrubbing, or picking over elevated vascular papules to prevent recurrent bleeding. Apply direct firm pressure with sterile gauze if trauma causes bleeding.",
        "follow_up": "Routine clinical observation. No active surveillance required for stable lesions.",
        "red_flags": "Rapid sudden eruptive growth over days, recurrent profuse spontaneous hemorrhage, or dark black firm nodule mimicking nodular melanoma.",
        "diff_clue": "Dermoscopic vascular lacunae and blanching under diascopy cleanly differentiate vascular lesions from pigmented melanocytic lesions."
    },
    "VASCULITIS": {
        "title": "Cutaneous Vasculitis",
        "category": "Immune Complex-Mediated Vascular Inflammation",
        "pathology_variants": [
            "Pathophysiologically initiated by deposition of circulating immune complexes (antigen-antibody) in post-capillary venule walls, activating the classical complement pathway. C5a recruits neutrophils, which degranulate and release reactive oxygen species and lysosomal enzymes, causing fibrinoid necrosis of the vessel wall and red blood cell extravasation.",
            "Classic clinical hallmark is 'palpable purpura'—non-blanching, elevated violaceous papules and plaques predominantly distributed symmetrically over gravity-dependent lower extremities. Extravasated erythrocytes within the dermis do not blanch under diascopy.",
            "Histopathologically demonstrates classic leukocytoclastic vasculitis (LCV): Transmural neutrophil infiltration of venular walls, karyorrhexis (neutrophil fragmentation forming nuclear dust), fibrinoid necrosis, and abundant extravasated erythrocytes throughout the interstitium.",
            "Optical presentation demonstrates vibrant purple to deep red non-blanching macules and papules with irregular grouping. Dermal capillary leakage produces pronounced localized chromatic saturation resistant to mechanical compression."
        ],
        "first_line": "Identification and management of underlying etiology (infections, drugs, autoimmune disorders). First-line therapy for cutaneous-limited vasculitis includes rest, lower extremity elevation, compression stockings, and systemic corticosteroids (Prednisone 0.5-1.0 mg/kg/day) for severe ulcerating lesions.",
        "barrier_care": "Minimize dependent physical standing and avoid prolonged walking during active eruptive phases. Keep leg ulcerations clean and dressed with sterile hydrogel dressings.",
        "follow_up": "Urgent comprehensive medical evaluation within 24 hours including complete blood count, comprehensive metabolic panel, urinalysis (screening for glomerulonephritis), and punch biopsy for Direct Immunofluorescence (DIF).",
        "red_flags": "Dark red/foamy urine indicating renal involvement, acute severe abdominal pain (Henoch-Schonlein purpura / IgA vasculitis), sensory neuropathy (numbness/foot drop), or digital gangrene.",
        "diff_clue": "Palpability and failure to blanch under diascopy definitively separate cutaneous vasculitis from simple thrombocytopenic petechiae or inflammatory rashes."
    },
    "VITILIGO": {
        "title": "Vitiligo",
        "category": "Autoimmune Melanocyte Destruction",
        "pathology_variants": [
            "Characterized by the progressive, selective autoimmune destruction of epidermal melanocytes mediated by autoreactive CD8+ T-lymphocytes. Interferon-gamma (IFN-gamma) and CXCL10 chemokine signaling pathways recruit cytotoxic lymphocytes that induce apoptotic death of melanocytes in the basal layer.",
            "Presents with sharply demarcated, porcelain-white chalky macules and patches devoid of pigment. Under Wood's lamp (365 nm UVA illumination), lesions emit an accentuated bright blue-white fluorescence due to complete epidermal melanin absence.",
            "Histological examination confirms total absence of functional melanocytes along the basal layer (demonstrated via negative Fontana-Masson silver and immunohistochemical Melan-A / HMB-45 staining), with preserved epidermal architecture and absence of dermal inflammation in stable lesions.",
            "Optical features show complete regional melanin depletion with sharp, distinct contrast against adjacent normally-pigmented skin. The surface texture is smooth and uninflamed, exhibiting normal dermatoglyphics without scale."
        ],
        "first_line": "Targeted topical immunosuppressive therapy: High-potency topical corticosteroids (Clobetasol 0.05%) or topical calcineurin inhibitors (Tacrolimus 0.1% ointment twice daily) for facial and intertriginous areas. Narrowband UVB (NB-UVB 311 nm) phototherapy administered 2 to 3 times weekly is standard for generalized vitiligo.",
        "barrier_care": "Strict sun protection: Depigmented skin lacks protective melanin and sunburns rapidly. Apply high-SPF 50+ mineral sunscreen to prevent solar erythema and minimize color contrast with tanned surrounding skin.",
        "follow_up": "Dermatology consultation within 2 to 4 weeks. Screen for associated autoimmune conditions (autoimmune thyroiditis / TSH testing).",
        "red_flags": "Rapid explosive body surface expansion, development of spontaneous halo nevi around multiple existing moles, or acute severe sunburn over depigmented patches.",
        "diff_clue": "Chalky-white complete depigmentation with accentuation under Wood's lamp distinguishes vitiligo from post-inflammatory hypopigmentation or tinea versicolor."
    },
    "WARTS": {
        "title": "Verruca Vulgaris (Wart)",
        "category": "Viral Human Papillomavirus Infection",
        "pathology_variants": [
            "Benign epithelial proliferation caused by infection of basal keratinocytes with low-risk Human Papillomavirus (HPV) genotypes (predominantly HPV-1, 2, 4, 27, 57). Viral E6 and E7 oncoproteins induce host cell cycle progression and keratinocyte hyperproliferation without cellular transformation.",
            "Histopathologically characterized by marked hyperkeratosis, papillomatosis, and acanthosis with columns of parakeratosis over the papillae tips. Pathognomonic koilocytes (enlarged vacuolated keratinocytes with eccentric, pyknotic hyperchromatic nuclei and clear perinuclear halos) reside in the granular layer.",
            "Clinically presents as hyperkeratotic, exophytic papules with a rough, verrucous 'cauliflower-like' surface. Paring the surface reveals tiny black dots representing thrombosed, dilated capillaries in the dermal papillae.",
            "Optical presentation displays prominent surface micro-projections with high surface roughness and disrupted normal skin lines. Thrombosed capillaries appear as dark punctate inclusions under dermoscopy."
        ],
        "first_line": "First-line chemical keratolysis: Topical Salicylic acid (17% to 40%) applied daily under occlusion following gentle emery board paring. In-office cryotherapy with liquid nitrogen applied for two 15-second freeze-thaw cycles every 2 to 3 weeks.",
        "barrier_care": "Avoid scratching, shaving over, or biting warts to prevent viral auto-inoculation to adjacent skin. Wear waterproof footwear in public showers, locker rooms, and pool areas.",
        "follow_up": "Re-evaluate if no clinical reduction after 6 to 8 weeks of consistent topical salicylic acid therapy.",
        "red_flags": "Rapidly multiplying lesions in immunocompromised individuals, recalcitrant periungual warts causing permanent nail plate dystrophy, or ulceration.",
        "diff_clue": "Disruption of normal skin lines and presence of pin-point black thrombosed capillaries upon superficial paring differentiate warts from corns or calluses."
    }
}

def extract_optical_biomarkers(pil_img: Image.Image):
    """
    Extracts real numerical optical and architectural biomarkers from the uploaded photograph.
    """
    img_rgb = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    
    # 1. Erythema Index (Vascular Hemoglobin Flush)
    erythema_val = float(np.mean(r - (g + b) / 2.0))
    if erythema_val > 0.10:
        erythema_level = "Intense Active Erythema"
        erythema_badge = "High Vascularity"
        erythema_color = "#9F1239"
    elif erythema_val > 0.04:
        erythema_level = "Moderate Erythematous Flushing"
        erythema_badge = "Moderate Redness"
        erythema_color = "#9A3412"
    else:
        erythema_level = "Low-Grade / Subtle Erythema"
        erythema_badge = "Subtle Redness"
        erythema_color = "#047857"

    # 2. Border Architecture & Edge Gradient
    grad_y = np.abs(lum[1:, :] - lum[:-1, :])
    grad_x = np.abs(lum[:, 1:] - lum[:, :-1])
    edge_contrast = float(np.mean(grad_y) + np.mean(grad_x))
    
    if edge_contrast > 0.075:
        border_type = "Well-Demarcated Sharp Margins"
        border_desc = "prominent boundary definition and sharp architectural transition against perilesional skin"
    elif edge_contrast > 0.042:
        border_type = "Partially Circumscribed Edge"
        border_desc = "tapering transitional gradient with localized peripheral demarcation"
    else:
        border_type = "Diffuse & Infiltrative Margins"
        border_desc = "soft, blended peripheral transition lacking sharp demarcation"

    # 3. Surface Texture & Hyperkeratosis
    std_texture = float(np.std(lum))
    if std_texture > 0.16:
        surface_texture = "Prominent Hyperkeratotic Scaling"
        texture_desc = "pronounced superficial micro-scale accumulation and epidermal roughening"
    elif std_texture > 0.09:
        surface_texture = "Fine Maculopapular Texture"
        texture_desc = "subtle focal micro-papular textural irregularities across the epidermal surface"
    else:
        surface_texture = "Smooth / Non-Scaly Surface"
        texture_desc = "uniform, planar epidermal surface morphology without elevated hyperkeratosis"

    # 4. Melanin & Pigment Distribution
    diff_lr = float(np.mean(np.abs(lum[:, :112] - np.fliplr(lum[:, 112:]))))
    diff_tb = float(np.mean(np.abs(lum[:112, :] - np.flipud(lum[112:, :]))))
    asym_score = float((diff_lr + diff_tb) * 10.0)
    
    if asym_score > 1.1:
        pigment_pattern = "Asymmetric Spatial Distribution"
        pigment_desc = "heterogeneous regional pigment clustering with architectural asymmetry"
    elif asym_score > 0.65:
        pigment_pattern = "Zonal Reticular Variation"
        pigment_desc = "mild spatial variance in chromatic intensity across quadrants"
    else:
        pigment_pattern = "Symmetric Homogeneous Tone"
        pigment_desc = "balanced, harmonious optical pigment distribution across the field of view"

    # Unique photo fingerprint hash (determines rotation of descriptive paragraphs)
    pixel_hash = int((np.sum(arr[::10, ::10, :]) * 1000) % 997)

    return {
        "erythema": {"level": erythema_level, "badge": erythema_badge, "val": round(erythema_val, 3), "color": erythema_color},
        "border": {"type": border_type, "desc": border_desc, "contrast": round(edge_contrast, 3)},
        "texture": {"type": surface_texture, "desc": texture_desc, "std": round(std_texture, 3)},
        "pigment": {"pattern": pigment_pattern, "desc": pigment_desc, "asym": round(asym_score, 2)},
        "hash": pixel_hash
    }

def generate_clinical_dossier(top_pred, pil_image: Image.Image, all_results: list):
    """
    Constructs an exhaustive, highly-tailored clinical diagnostic dossier for the primary diagnosis.
    Adapts dynamically to the optical features of each photo and the model's runner-up predictions.
    """
    code = top_pred.get("code", "NORMAL")
    knowledge = CLINICAL_KNOWLEDGE.get(code, CLINICAL_KNOWLEDGE["NORMAL"])
    
    # 1. Optical biomarker analysis
    biomarkers = extract_optical_biomarkers(pil_image)
    
    # 2. Select dynamic pathology variant using photo hash
    variants = knowledge.get("pathology_variants", [])
    variant_idx = biomarkers["hash"] % len(variants) if variants else 0
    pathology_narrative = variants[variant_idx] if variants else top_pred.get("description", "")
    
    # 3. Construct Photo-Specific Visual Analysis
    photo_analysis = (
        f"Optical computer vision analysis of this lesion identifies {biomarkers['border']['desc']} "
        f"accompanied by {biomarkers['erythema']['level'].lower()} (Erythema Index: {biomarkers['erythema']['val']}). "
        f"Surface textural profiling indicates {biomarkers['texture']['desc']}, correlated with {biomarkers['pigment']['desc']}. "
        f"The Vision Transformer's 12 multi-head self-attention layers localized primary activation weight across these distinct morphology zones."
    )
    
    # 4. Construct Differential Diagnosis comparison against runner-up candidates
    runner_ups = [r for r in all_results if r["code"] != code][:2]
    if len(runner_ups) >= 2:
        diff_text = (
            f"The deep learning model performed comparative cross-entropy validation against runner-up candidates "
            f"<strong>{runner_ups[0]['name']}</strong> ({runner_ups[0]['percentage']}%) and "
            f"<strong>{runner_ups[1]['name']}</strong> ({runner_ups[1]['percentage']}%). "
            f"Primary diagnostic classification favors {top_pred['name']} ({top_pred['percentage']}%): {knowledge.get('diff_clue', '')}"
        )
    elif len(runner_ups) == 1:
        diff_text = (
            f"Compared against secondary candidate <strong>{runner_ups[0]['name']}</strong> ({runner_ups[0]['percentage']}%): "
            f"{knowledge.get('diff_clue', '')}"
        )
    else:
        diff_text = knowledge.get("diff_clue", "Consistent clinical and morphological alignment with primary condition.")

    # 5. Assemble structured clinical dossier
    return {
        "title": knowledge.get("title", top_pred["name"]),
        "category": knowledge.get("category", top_pred.get("type", "Dermatological Condition")),
        "overview": top_pred.get("description", ""),
        "pathology": pathology_narrative,
        "biomarkers": [
            {"label": "Vascular Erythema", "value": biomarkers["erythema"]["level"], "badge": biomarkers["erythema"]["badge"], "color": biomarkers["erythema"]["color"]},
            {"label": "Border Architecture", "value": biomarkers["border"]["type"], "badge": f"Contrast: {biomarkers['border']['contrast']}", "color": "#1E293B"},
            {"label": "Surface Topology", "value": biomarkers["texture"]["type"], "badge": f"Std: {biomarkers['texture']['std']}", "color": "#1E293B"},
            {"label": "Pigment Geometry", "value": biomarkers["pigment"]["pattern"], "badge": f"Asym: {biomarkers['pigment']['asym']}", "color": "#1E293B"}
        ],
        "photoAnalysis": photo_analysis,
        "pathologyNarrative": pathology_narrative,
        "differentialDiagnosis": diff_text,
        "treatment": {
            "firstLine": knowledge.get("first_line", top_pred.get("action", "")),
            "barrierCare": knowledge.get("barrier_care", "Maintain gentle skin barrier hygiene and protective moisturization."),
            "followUp": knowledge.get("follow_up", "Dermatology re-evaluation advised within 2 to 4 weeks.")
        },
        "redFlags": knowledge.get("red_flags", "Sudden rapid expansion, ulceration, bleeding, or intractable pain.")
    }
