import os
from PIL import Image
import numpy as np

#BASE_DIR = os.path.dirname(__file__)
#IMAGE_DIR = os.path.join(BASE_DIR, "images")

OPTIONS = {
    "Option A": {"value": "A", "image": "a.png"},
    "Option B": {"value": "B", "image": "b.png"},
    "Option C": {"value": "C", "image": "c.png"},
}

OUTFITS = {
            "o0001": "layered chiffon blouse with delicate pleats, soft ruffled cuffs, subtle pearl buttons, paired with slim high-waisted trousers",
            "o0002": "oversized wool sweater with thick cable-knit texture, loose turtleneck fold, paired with fitted leggings",
            "o0003": "sleek satin slip dress with thin straps, soft draping across the chest, subtle sheen along the fabric",
            "o0004": "structured leather jacket with silver zippers, cropped fit, worn over a minimalist cotton tank top",
            "o0005": "high-waisted pleated skirt with flowing movement, paired with a fitted ribbed top",
            "o0006": "layered streetwear outfit with baggy cargo pants, cropped hoodie, and utility belt",
            "o0007": "elegant evening gown with embroidered lace bodice, long sheer sleeves, and flowing silk skirt",
            "o0008": "lightweight summer dress with floral patterns, puffed sleeves, and smocked waistline",
            "o0009": "sporty outfit with compression leggings, racerback sports bra, and breathable mesh jacket",
            "o00010": "denim jacket with frayed hem, white tee underneath, paired with black skinny jeans",
            "o00011": "flowy bohemian maxi dress with layered fabrics, beaded embroidery, and braided belt",
            "o00012": "formal blazer with sharp shoulders, satin lapels, tailored trousers, minimalistic blouse",
            "o00013": "knit cardigan with large wooden buttons, soft texture, worn over a simple linen dress",
            "o00014": "black cocktail dress with asymmetrical hemline, fitted waist, subtle glitter accents",
            "o00015": "high-fashion ensemble with oversized trench coat, silk scarf, tailored pencil skirt",
            "o00016": "romantic lace top with scalloped edges, layered camisole, paired with soft tulle skirt",
            "o00017": "casual outfit with cropped tee, wide-leg jeans, and canvas sneakers",
            "o00018": "futuristic outfit with metallic bodysuit, geometric cutouts, reflective material",
            "o00019": "soft knit dress with long sleeves, belt around the waist, warm cozy fabric",
            "o00020": "velvet jumpsuit with deep V neckline, flared legs, subtle embroidered trims",
            "o00021": "streetwear combo with oversized bomber jacket, graphic tee, and straight-leg jeans",
            "o00022": "sleek business dress with fitted silhouette, structured seams, minimalist belt",
            "o00023": "winter outfit with puffer jacket, thick scarf, thermal leggings, fur-trimmed boots",
            "o00024": "delicate silk blouse with bishop sleeves, tucked into high-waisted tailored shorts",
            "o00025": "ruffled mini dress with layered skirt, soft pastel tones, ribbon details",
            "o00026": "layered goth-inspired outfit with black corset top, mesh sleeves, pleated mini skirt",
            "o00027": "retro 70s outfit with flared jeans, patterned blouse, wide leather belt",
            "o00028": "elegant kimono-style wrap dress with wide sleeves and embroidered sash",
            "o00029": "minimalist outfit with monochrome turtleneck, straight trousers, clean lines",
            "o00030": "summer set with cropped linen top, flowy midi skirt, straw accessories",
            "o00031": "denim overalls worn over a striped fitted tee, rolled-up pant legs",
            "o00032": "evening outfit with sequined top, slim black pants, and matching heels",
            "o00033": "soft cardigan layered over camisole, paired with high-waisted pleated shorts",
            "o00034": "tailored coat with double-breasted buttons, turtleneck dress underneath",
            "o00035": "retro pin-up outfit with polka-dot dress, cinched waist, and ribbon hair bow",
            "o00036": "ballet-inspired outfit with wrap sweater, tulle skirt, and delicate slippers",
            "o00037": "punk-inspired fit with studded leather jacket, fishnet top, tartan skirt",
            "o00038": "sleek catsuit with matte fabric, zipper front, and defined seams",
            "o00039": "romantic outfit with off-shoulder blouse, layered pearl necklace, chiffon skirt",
            "o00040": "cozy fall outfit with plaid scarf, knit sweater dress, thick stockings",
            "o00041": "classy ensemble with silk blouse, pencil skirt, and minimal gold jewelry",
            "o00042": "activewear with crop top, biker shorts, and lightweight running jacket",
            "o00043": "evening outfit with velvet blazer, satin camisole, and slim-fit trousers",
            "o00044": "chic dress with square neckline, puff sleeves, and structured bodice",
            "o00045": "casual lounge set with oversized sweatshirt and soft jogger pants",
            "o00046": "romantic vintage outfit with lace gloves, floral dress, pearl earrings",
            "o00047": "festival outfit with fringed vest, denim shorts, and crochet top",
            "o00048": "formal gown with draped neckline, fitted bodice, flowing train",
            "o00049": "sleek minimalist bodysuit paired with wide-leg tailored pants",
            "o00050": "warm outfit with shearling jacket, wool skirt, and thick tights",
            "o00051": "cute outfit with cropped hoodie, plaid mini skirt, and knee-high socks",
            "o00052": "street style with bucket hat, oversized flannel, baggy jeans",
            "o00053": "glamorous party dress with shimmering fabric, high slit, fitted waist",
            "o00054": "classy trench coat over silky slip top and tailored cigarette pants",
            "o00055": "Renaissance-inspired blouse with puff sleeves and laced corset bodice",
            "o00056": "modern outfit with asymmetrical top, structured skirt, clean minimal lines",
            "o00057": "soft pastel sweater with pleated midi skirt and delicate jewelry",
            "o00058": "rock-inspired leather pants, graphic tank top, studded belt",
            "o00059": "cool outfit with cropped bomber jacket, joggers, and chunky sneakers",
            "o00060": "elegant blouse with draped scarf collar, high-waisted palazzo pants",
            "o00061": "romantic top with floral embroidery, soft tulle mini skirt",
            "o00062": "all-black outfit with ribbed turtleneck, ankle boots, straight-leg pants",
            "o00063": "evening dress with ornate beading, sheer sleeves, soft flowing skirt",
            "o00064": "summer outfit with off-shoulder crop top, high-waisted shorts",
            "o00065": "sleek bodysuit with cutout details, paired with suede skirt",
            "o00066": "soft cardigan and pleated maxi skirt with earthy tones",
            "o00067": "classic blouse with ruffled collar, tailored trousers, leather loafers",
            "o00068": "party outfit with metallic mini skirt, mesh top, and bold accessories",
            "o00069": "winter coat with faux fur trim, knit dress, thick scarf",
            "o00070": "chic jumpsuit with belted waist and structured shoulders",
            "o00071": "vintage denim jacket, pastel tee, and floral skirt",
            "o00072": "silky halter dress with low back and subtle drape",
            "o00073": "minimal street fit with cropped tank, cargo pants, chain belt",
            "o00074": "luxury outfit with faux-fur coat, glitter top, sleek leggings",
            "o00075": "classic wrap dress with soft folds and tied waist",
            "o00076": "techwear outfit with layered straps, waterproof jacket, tapered pants",
            "o00077": "cozy knit set with cropped sweater and matching midi skirt",
            "o00078": "romantic dress with tiered ruffles and embroidered bodice",
            "o00079": "sleek fashion suit with fitted blazer and straight trousers",
            "o00080": "light summer blouse with bow tie collar and wide linen pants",
            "o00081": "streetstyle puffer vest, thermal top, cargo joggers",
            "o00082": "corset top with lace detailing, layered chiffon skirt",
            "o00083": "soft romper with delicate straps and ruffled hems",
            "o00084": "formal office dress with structured seams and belt detail",
            "o00085": "punk mini dress with chains, mesh sleeves, and platform boots",
            "o00086": "comfortable outfit with oversized tee and cotton leggings",
            "o00087": "sleek monochrome dress with high neckline and clean silhouette",
            "o00088": "romantic pastel outfit with bow cardigan and lace skirt",
            "o00089": "evening set with glitter crop top and high-waisted maxi skirt",
            "o00090": "athleisure outfit with hooded crop jacket, tight leggings",
            "o00091": "gothic dress with velvet panels, lace sleeves, and choker",
            "o00092": "vintage-inspired cardigan, satin blouse, and A-line skirt",
            "o00093": "summer romper with floral pattern and fitted waist",
            "o00094": "luxury satin blouse with draped neckline and long skirt",
            "o00095": "street outfit with oversized hoodie and denim cutoffs",
            "o00096": "sleek strapless gown with structured bodice and long train",
            "o00097": "layered outfit with denim vest, ribbed top, and cargo skirt",
            "o00098": "cozy wool coat with high collar, knit sweater, leggings",
            "o00099": "minimal geometric dress with clean lines and simple accents",
            "o0100": "romantic embroidered blouse paired with flowing chiffon trousers"
}

class PromptLibOutfits:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selection": (
                    list(OUTFITS.keys()),
                    {"default": "o0001"}
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Rückgabewert ignoriert selection → Downstream bleibt stabil
        """
        return False

    def run(self, selection):
        #option = OPTIONS[selection]
        #value = option["value"]
        value = OUTFITS[selection]
        return (value, )