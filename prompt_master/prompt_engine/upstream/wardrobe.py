"""
wardrobe.py — Claude Prompt LD
Seed-picked wardrobe bias so the LLM doesn't default to "a simple dress."

Axes per seed: piece · color · fabric · worn detail · (optional) footwear.
Banks are direction only — never script words the model must copy.

Bias is skipped when the intent already names clothing (or underwear suite
takes over). Heat-gated DARING banks only enter when the intent reads explicit.

Laws (Grok-style / LTX-literal):
  - Clothes are physical objects: cut, cling, gap, travel with the body
  - Re-assert how they sit across beats (not one fashion line then forgotten)
  - Undress is grip → pull → reveal → resting state, never a vanish
  - No "sexy/alluring" adjectives — show why via fabric and fit
"""

import random
import re

_NAMED_CLOTHES = re.compile(
    r"(?i)\b(dress(?!\s+(?:party|code|down|up\b))|skirt|gown|blouse|shirt|jeans|shorts|leggings?|bra|panties|"
    r"lingerie|corset|stockings?|robe|coat|jacket|sweater|hoodie|bikini|"
    r"swimsuit|uniform|slip|camisole|bodysuit|jumpsuit|romper|crop\s*top|"
    r"tank|tee|t-shirt|suit|trousers|pants|heels?|boots?|wearing|dressed|"
    r"outfit|topless|naked|nude|towel|apron|kimono|thong|nightie|negligee|"
    r"teddy|garter|bustier|catsuit|harness|onesie|pajamas?|pyjamas?|leotard|tutu|costume|cosplay|cheer\s*outfit)\b"
)

_EXPLICIT = re.compile(
    r"(?i)(?<![a-z])(fuck\w*|cock|pussy|cum\w*|blowjob|handjob|anal|"
    r"penetrat\w*|thrust\w*|orgasm\w*|nipples?|nude|naked|topless|nsfw|"
    r"erotic|strip\w*|undress\w*|tits?|breasts?|sex\w*|porn|grind\w*|"
    r"lap\s*dance|tease|seduc\w*)(?![a-z])"
)

# ═══ HER — dressed lanes ═════════════════════════════════════════════════════
HER_PIECES = [
    # everyday
    "a slip dress", "a wrap dress with a deep tie", "a ribbed knit dress",
    "a sundress with a full hem", "a shirt dress with the belt cinched",
    "a t-shirt dress and high socks", "a sweater dress hugging the hips",
    "a denim pinafore over a fitted tee", "a turtleneck tucked into a midi skirt",
    "an oversized tee worn as a dress", "a cropped cardigan and high-waist skirt",
    "a halter top and low-slung jeans", "an off-shoulder sweater and denim cutoffs",
    "a tube top and cargo pants", "a graphic tee knotted over a tennis skirt",
    "a flannel tied at the waist over a bodysuit", "mom jeans and a baby tee",
    "a tank tucked into paperbag shorts", "overalls with one strap hanging",
    "a hoodie half-zipped over a sports bra", "a peasant blouse and long skirt",
    "a polo dress with the collar popped", "low-rise flares and a shrunken cardigan",
    # going out
    "a bodycon mini in a bold print", "a backless top and leather mini",
    "a corset over tailored trousers", "a sequined party top and shorts",
    "a velvet slip gown", "a satin cowl-neck midi", "a bandage dress",
    "a blazer with nothing beneath it", "a vintage jumpsuit unzipped low",
    "a halter gown with a thigh slit", "a chainmail-look top and cigarette pants",
    "a bustier and wide-leg trousers", "a one-shoulder cocktail dress",
    "a metallic wrap top and leather leggings", "a fringe dress made to move",
    "a mesh long-sleeve over a solid bra", "a button-down tied at the waist and a micro skirt",
    "opera gloves with a strapless column dress", "a two-piece skirt set with a cutout waist",
    # work / roleplay-adjacent
    "a pencil skirt and a blouse straining one button", "a secretary blouse and pleated mini",
    "a lab coat over a fitted dress", "a chef's whites unbuttoned at the collar",
    "a bartender's vest and rolled sleeves", "a flight-attendant style skirt suit",
    "a waitress apron over a short uniform dress", "a schoolgirl-style plaid skirt and cropped sweater (adult)",
    "gym-teacher track pants and a tucked polo", "a nurse-style button dress (adult costume)",
    "a maid-style dress with an apron bow (adult costume)", "a business suit with nothing under the blazer",
    # lounge / sleep
    "a satin robe barely tied", "a camisole and boy-shorts",
    "a babydoll nightie", "silk pyjamas with the top misbuttoned",
    "an oversized band tee and ankle socks", "a waffle-knit lounge set",
    "a slip nightgown with lace at the hem", "a kimono robe over a matching set",
    "a cropped hoodie and cotton sleep shorts", "just his shirt, sleeves past her hands",
    # swim / beach
    "a string bikini under an open linen shirt", "a high-cut one-piece",
    "a triangle bikini top and a sarong", "a bandeau bikini and cutoff shorts",
    "a crochet cover-up over a dark bikini", "a sporty two-piece and surf shorts",
    "a plunge one-piece with a low back", "a wet swimsuit and a towel over one shoulder",
    # athletic
    "a sports bra and track pants", "matching seamless leggings and bra",
    "a tennis dress with built-in shorts", "bike shorts and a cropped windbreaker",
    "a boxing wrap top and satin shorts",
    "yoga flares and a strappy back bra", "a swim-team one-piece and slides",
    # dance / performance
    "a scoop-back leotard and loose leg warmers", "a high-cut leotard over shimmer tights",
    "a wrap ballet skirt over a camisole leotard", "a velvet long-sleeve leotard",
    "a figure-skating dress with an illusion neckline", "a tutu over a plain black leotard",
    "a rhythmic-gymnastics leotard in a swirl print", "a jazz cut-out unitard",
    "a cheer shell and pleated cheer skirt", "a pom squad crop and flip skirt",
    "a drill-team sequin leotard", "a pole-fitness set with grip shorts",
    # gym / training
    "a scrunch-seam leggings set", "a racerback bra and booty shorts",
    "a one-shoulder sports bra and flared leggings", "a zip-front bra and joggers",
    "a stringer tank over a longline bra", "a tennis skirt and cropped quarter-zip",
    "a wrestling-style singlet (adult)", "compression shorts and a tied-up tee",
    "a boxing crop and satin trunks", "climbing tights and a chalky tank",
    # cosplay / costume archetypes (generic adult — no IP names)
    "a maid dress with cat-ear headband (adult costume)", "a devil-horns red mini set (adult costume)",
    "an angel-wings white slip set (adult costume)", "a bunny leotard with cuffs and collar (adult costume)",
    "a witch hat and laced black dress (adult costume)", "a vampire cape over a corset gown (adult costume)",
    "a sailor-collar pleated uniform (generic, adult)", "a gothic frill dress with platform boots",
    "a fantasy-elf drape dress with a leaf motif", "a superhero-style bodysuit with a bold emblem-free chest",
    "a race-queen umbrella-girl two-piece (adult costume)", "an idol stage outfit with a layered skirt",
    "a police-style shirt dress with a toy badge (adult costume)", "a pirate blouse and underbust corset (adult costume)",
    "a nun-style habit mini (adult costume)", "a cyber-street set with reflective strips",
    "a cat-girl leotard with tail and ears (adult costume)", "a fox-spirit shrine maiden set (adult costume)",
    "a succubus horn-and-garter set (adult costume)", "a fallen-angel harness and black wings (adult costume)",
    "a nurse latex-look mini with a toy cap (adult costume)", "a cheer captain crop and micro skirt (adult costume)",
    "a school swimsuit one-piece (adult costume)", "a magician's assistant sequin leotard (adult costume)",
    "a samurai-inspired wrap mini and obi belt (adult costume)", "a geisha-style kimono open at the thigh (adult costume)",
    "a cowgirl fringe top and denim cutoffs (adult costume)", "a pin-up sailor shorts set (adult costume)",
    "a space-girl silver unitard (adult costume)", "a neon rave kandi and mesh set (adult costume)",
    "a jester diamond-harlequin leotard (adult costume)", "a queen crown and velvet robe over lingerie (adult costume)",
    "a knight breastplate over a sheer underdress (adult costume)", "a dragon-scale bodycon (adult costume)",
    "a mermaid sheer-scale skirt and shell top (adult costume)", "a fairy petal micro dress and wing harness (adult costume)",
    "a mummy wrap strips over a nude base (adult costume)", "a skeleton-print bodysuit cut high (adult costume)",
    "a red-riding hood cape over a corset mini (adult costume)", "a steampunk bustier and goggles (adult costume)",
    "a street-fighter crop and high socks (adult costume)", "a magical-girl bow and pleated micro skirt (adult costume)",
    "a shrine-maiden red-white layered set (adult costume)", "a demon-hunter leather straps and shorts (adult costume)",
    "a princess ballgown unzipped low (adult costume)", "a villainess catsuit with a high collar (adult costume)",
    "a cheerleading varsity crop and pleats (adult costume)", "a flight-attendant scarf and pencil skirt (adult costume)",
    "a racing jacket zipped over a bikini (adult costume)", "a yoga-instructor matching set worn sheer (adult costume)",
    # statement / sexy street
    "a latex-look midi dress", "a vinyl trench with little under it",
    "a corset laced over a sheer blouse", "leather trousers and a silk scarf top",
    "a slip dress under a chunky knit falling off one shoulder",
    "a cheongsam-style mini", "a velvet blazer dress with a plunge",
    "fishnet layers under a solid slip", "a body-chain worn over a plain black dress",
    "a wet-look midi with a thigh split", "a strappy bondage-style dress (fashion, adult)",
    "a cutout bodycon with underboob window", "a sling dress barely covering",
    "a metallic micro two-piece", "a backless maxi with a chain strap",
    "a sheer black blouse over a black bra and leather skirt", "a lace catsuit zipped to the navel",
]

# ═══ HER — daring lanes (explicit intent only) ═══════════════════════════════
HER_DARING = [
    # lingerie as the outfit
    "a lace bralette and matching thong", "a strappy cage bra set",
    "a sheer teddy with nothing under it", "a satin chemise riding high",
    "a quarter-cup bra and high-waist knickers", "a garter belt, stockings, and little else",
    "a bodystocking in open fishnet", "a corset, garters, and seamed stockings",
    "an unlined mesh bra and cheeky briefs", "a plunge teddy cut to the navel",
    "a negligee that stops mid-thigh, gaping", "a bridal-white lace set with a garter",
    "a leather harness over bare skin", "a latex bra and micro skirt",
    "a peekaboo cutout set", "crotchless lace and a silk robe open over it",
    "a G-string and an unbuttoned men's shirt", "pasties and low-rise briefs",
    "an open-cup bra and garter straps only", "a waist cincher and sheer briefs",
    "a strappy body harness and thigh garters", "a sheer babydoll with no panties",
    "a pearl body chain and a thong", "a lace robe fully open over a G-string",
    "a micro bikini as indoor lingerie", "a wet white shirt and nothing under",
    # bare / near-bare states
    "topless in low-rise jeans", "nothing but a gold body chain",
    "just a towel knotted at the chest", "bare under a fur coat held closed",
    "only stockings and heels", "an apron and nothing behind it",
    "wet lingerie gone transparent", "just panties and crossed arms",
    "a sheet wrapped and slipping", "oiled skin and a micro bikini",
    "one of his ties and nothing else", "fully nude with jewelry only",
    "just heels and a choker", "a blazer held closed over bare skin",
    "his hoodie zipped once over bare everything", "only a garter belt and stockings",
    # club / show / strip
    "a micro dress that reads as a long top", "a wet-look catsuit unzipped to the sternum",
    "rhinestone pasties under an open mesh top", "a dancer's rhinestone two-piece and thigh boots",
    "a cupless corset and tulle skirt", "a chain bikini",
    "sheer everything over a thong silhouette", "a bunny-style leotard and cuffs (adult costume)",
    "a tearaway skirt over a rhinestone thong", "pasties and a G-string on a stage",
    "a oil-slick body and tiny briefs", "a cage dress with nothing under",
    # sexy cosplay (explicit-only heat)
    "a bunny leotard with the crotch cut high and open back (adult costume)",
    "a maid dress unbuttoned to the navel over a garter set (adult costume)",
    "a nurse mini with garters and no hose (adult costume)",
    "a schoolgirl plaid skirt and a tied-up white shirt with no bra (adult costume)",
    "a cat-girl set: ears, tail plug-look harness, and lingerie (adult costume)",
    "a succubus lingerie set with horns and garter wings (adult costume)",
    "a nun habit open over a black lace set (adult costume)",
    "a police shirt unbuttoned over a thong and badge necklace (adult costume)",
    "a cheer crop pulled up and a skirt with nothing under (adult costume)",
    "a race-queen two-piece reduced to pasties and bottoms (adult costume)",
    "a shrine-maiden top half-off and red skirt hiked (adult costume)",
    "a magical-girl leotard sheer and wet (adult costume)",
]

# ═══ HIM — dressed lanes ═════════════════════════════════════════════════════
HIS_PIECES = [
    # everyday
    "an open flannel over a bare chest", "a fitted black tee and jeans",
    "a linen shirt with rolled sleeves", "a henley pushed up at the forearms",
    "a denim jacket over a white tee", "a crewneck and worn chinos",
    "a polo with the collar soft", "a hoodie and relaxed jeans",
    "a bomber over a plain tank", "an overshirt hanging open",
    "a rugby shirt and joggers", "a chore coat and straight-leg trousers",
    # dressed up
    "a half-unbuttoned dress shirt and slacks", "a tuxedo shirt undone at the collar",
    "a three-piece with the jacket shed", "a turtleneck and tailored coat",
    "a silk shirt open two buttons", "a waistcoat over rolled shirtsleeves",
    "a sharp suit with the tie pulled loose", "a band-collar shirt and pleated trousers",
    # work / roleplay-adjacent
    "work trousers and a tool-worn tank", "a mechanic's coverall tied at the waist",
    "a chef's apron over a fitted tee", "a firefighter's suspenders over a snug shirt (adult costume)",
    "a doctor's coat over scrubs (adult costume)", "a security shirt straining the shoulders",
    "carpenter jeans, no shirt, a pencil behind the ear", "a bartender's rolled sleeves and open vest",
    # lounge / athletic
    "grey joggers and nothing else", "a silk robe hanging open",
    "gym shorts and taped hands", "compression top and track pants",
    "a track jacket half-zipped", "basketball shorts and a backwards cap",
    "swim trunks and a towel around the neck", "pyjama pants slung low",
    # costume / performance / cosplay (generic adult — no IP names)
    "a pirate shirt half-laced (adult costume)", "a vampire collar cloak over bare chest (adult costume)",
    "a superhero-style suit top, emblem-free", "a firefighter costume with suspenders down (adult costume)",
    "a gladiator-style leather skirt and straps (adult costume)", "a stage magician's vest and rolled cuffs",
    "a wrestling singlet pulled to the waist (adult)", "a lifeguard tank and red shorts",
    "a knight chest plate over bare abs (adult costume)", "a demon-horns harness and leather pants (adult costume)",
    "a cat-boy ears and open silk shirt (adult costume)", "a vampire hunter coat over a bare chest (adult costume)",
    "a samurai wrap top open at the chest (adult costume)", "a cowboy hat and open denim vest (adult costume)",
    "a butler waistcoat with no shirt (adult costume)", "a Roman toga draped low (adult costume)",
    "a cyber-street mesh top and cargo pants", "a wrestler entrance robe open (adult costume)",
    "a mafia suit with the shirt half open", "a rockstar leather pants and no shirt",
    "a construction-site hard-hat and tool belt over a bare chest (adult costume)",
    "a paramedic jacket open over a tight tee (adult costume)", "a bartender apron and bare chest",
    # statement
    "a leather jacket over a white tank", "an unsnapped western shirt",
    "a vest with no shirt beneath", "a denim jacket over a bare torso",
    "a mesh tank and leather trousers", "an open Cuban-collar shirt in a loud print",
    "a hoodie unzipped over bare skin", "a peacoat over a bare chest",
    "a chain necklace and open flannel", "a silk robe and nothing else visible",
    "a suit jacket over bare skin and slacks", "a towel around the neck and low shorts",
]

# ═══ HIM — daring lanes (explicit intent only) ═══════════════════════════════
HIS_DARING = [
    "boxer briefs and nothing else", "a towel low on the hips",
    "an apron and bare everything behind it", "just unbuttoned jeans",
    "a leather harness and dark jeans", "briefs and an open silk robe",
    "fully nude, dog tags only", "wet trunks clinging low",
    "a jockstrap and gym socks", "an unzipped wetsuit peeled to the waist",
    "suit trousers, belt open, shirt gone", "nothing but a sheet at the waist",
    "only a bowtie and dress socks (adult formal joke)", "a jockstrap and open flannel",
    "briefs and a loosened tie only", "grey sweatpants with a clear outline, no shirt",
    "a towel falling open", "leather chaps over a jockstrap (adult costume)",
    "a firefighter pant with suspenders and bare chest (adult costume)",
    "a gladiator skirt and nothing under (adult costume)",
    "a wrestler singlet peeled to the hips (adult)", "a robe open over full nude",
    "jeans shoved to mid-thigh", "only a belt and open shirt",
]

FABRICS = [
    "silk", "satin", "washed cotton", "rib knit", "worn denim", "soft leather",
    "lace", "sheer mesh", "crushed velvet", "linen", "jersey", "latex",
    "chiffon", "brushed flannel", "fishnet", "wet-look vinyl", "cashmere",
    "sequin", "tulle", "suede", "terry cloth", "corduroy", "organza",
    "stretch scuba", "crochet", "faux fur", "modal", "raw silk", "boiled wool",
    "nylon sheen", "cotton poplin", "spandex blend", "crinkle gauze",
]

COLORS = [
    "ink black", "bone white", "oxblood", "champagne", "dusty rose",
    "midnight blue", "emerald", "blood red", "smoke grey", "burnt caramel",
    "electric cobalt", "pearl", "matte olive", "hot pink", "lavender haze",
    "gold-shot", "silver", "deep plum", "butter yellow", "seafoam",
    "espresso brown", "neon lime", "faded lilac", "cherry cola",
]

DETAILS = [
    "one strap already fallen off the shoulder", "the hem riding where she last tugged it",
    "buttons open one too many", "the tie loosening on its own",
    "a zipper stopped halfway", "fabric still creased from sitting",
    "cling damp at the small of the back", "a sleeve pushed up past the elbow",
    "the waistband rolled once", "a hook left undone at the top",
    "static making it cling to one thigh", "the collar pulled off-center",
    "a seam sitting slightly twisted", "lace edge peeking past the neckline",
    "the slit finding more leg with each step", "a bra strap showing on purpose",  # HER-ONLY
    "sheer panels reading skin in the light", "the knot one shrug from giving",
    "sweat blooming faint between the shoulder blades", "a stocking top just visible",
    "the fabric worn thin where it stretches", "one cuff buttoned, one loose",
    "a thumb hooked in the waistband out of habit", "the back dipping lower than the front",
    "gooseflesh where the fabric ends", "a fresh press gone soft in the heat",
    "the drawstring in a lazy bow", "an outline showing through when the light crosses",
    "a tear starting small at the seam", "the shoulder seam sliding toward the arm",
    "elastic leaving a faint line on skin", "the wrap barely holding its overlap",
    "the neckline gapping when she leans forward", "a belt tip hanging free and swinging",
    "fabric bunching at the crook of the elbow", "the skirt twisting as the hips turn",
    "a cuff catching on a wrist", "the hem kicking with every step",
    "a strap mark fading on bare skin", "the placket pulling open mid-chest",
    "socks sagging at the ankle", "the jacket hanging off one shoulder only",
]

# Optional footwear lane — only when the seed rolls it in (keeps most shots uncluttered).
FOOTWEAR_HER = [
    "bare feet", "strappy heels", "ankle boots", "worn sneakers", "knee-high boots",
    "platform slides", "ballet flats kicked half-off", "thigh boots", "chunky loafers",
    "sandals with thin straps", "combat boots unlaced", "socks only",
]
FOOTWEAR_HIM = [
    "bare feet", "worn sneakers", "leather boots", "dress shoes, laces loose",
    "slides", "work boots", "socks only", "loafers, no socks",
]

# Shared physics law appended to every wardrobe / underwear block.
_CLOTHES_PHYSICS = (
    "CLOTHES ARE OBJECTS. Name each visible garment in plain words — never "
    "\"outfit\". State how it sits: cut, cling, gap, drape, what it frames. "
    "Clothes travel with the body; fabric answers motion (hem kick, strap "
    "slip, waistband ride). Establish the dressed state early, then re-assert "
    "fit later with short physical cues. Sexy, hot and alluring describe "
    "nothing the view can find — fabric, light and grip do."
)


_HER_CUES = re.compile(r"(?i)\b(she|her|hers|woman|girl|female|lady|wife|girlfriend|milf|babe|mistress)\b")
_HIM_CUES = re.compile(r"(?i)\b(he|him|his|man|guy|male|dude|husband|boyfriend|daddy|mister)\b")


def infer_who(intent: str) -> str:
    """Auto target: her-cues win (lead priority), then him-cues, default her."""
    t = intent or ""
    if _HER_CUES.search(t):
        return "her"
    if _HIM_CUES.search(t):
        return "him"
    return "her"


def wants_bias(intent: str) -> bool:
    return not _NAMED_CLOTHES.search(intent or "")


# ═══ OCCASION pools ══════════════════════════════════════════════════════════
# When the intent names a setting, bias her piece pool to fit it. Each occasion
# has a normal tier (always) and a spicy tier (explicit intent only). Detection
# is keyword-based; first match wins. No match → general HER_PIECES/HER_DARING.

_OCCASIONS = {
    "cosplay": {
        "cues": ("cosplay", "costume", "costume party", "halloween", "convention",
                 "con ", "comic con", "masquerade", "roleplay", "role play",
                 "cat girl", "catgirl", "bunny girl", "maid cafe", "anime",
                 "fantasy fair", "renaissance faire", "larp"),
        "normal": [
            "a bunny leotard with cuffs and collar (adult costume)",
            "a maid dress with cat-ear headband (adult costume)",
            "a sailor-collar pleated uniform (generic, adult)",
            "a witch hat and laced black dress (adult costume)",
            "a cat-girl leotard with tail and ears (adult costume)",
            "a fox-spirit shrine maiden set (adult costume)",
            "a succubus horn-and-garter set (adult costume)",
            "a nurse latex-look mini with a toy cap (adult costume)",
            "a race-queen umbrella-girl two-piece (adult costume)",
            "a magical-girl bow and pleated micro skirt (adult costume)",
            "a shrine-maiden red-white layered set (adult costume)",
            "a pirate blouse and underbust corset (adult costume)",
            "a steampunk bustier and goggles (adult costume)",
            "a cyber-street set with reflective strips",
            "a devil-horns red mini set (adult costume)",
            "an angel-wings white slip set (adult costume)",
            "a vampire cape over a corset gown (adult costume)",
            "a gothic frill dress with platform boots",
            "a space-girl silver unitard (adult costume)",
            "a jester diamond-harlequin leotard (adult costume)",
            "a red-riding hood cape over a corset mini (adult costume)",
            "a fairy petal micro dress and wing harness (adult costume)",
            "a cowgirl fringe top and denim cutoffs (adult costume)",
            "a police-style shirt dress with a toy badge (adult costume)",
        ],
        "spicy": [
            "a bunny leotard with the crotch cut high and open back (adult costume)",
            "a maid dress unbuttoned to the navel over a garter set (adult costume)",
            "a nurse mini with garters and no hose (adult costume)",
            "a schoolgirl plaid skirt and a tied-up white shirt with no bra (adult costume)",
            "a cat-girl set: ears, tail harness, and lingerie (adult costume)",
            "a succubus lingerie set with horns and garter wings (adult costume)",
            "a nun habit open over a black lace set (adult costume)",
            "a police shirt unbuttoned over a thong (adult costume)",
            "a cheer crop pulled up and a skirt with nothing under (adult costume)",
            "a race-queen two-piece reduced to pasties and bottoms (adult costume)",
            "a magical-girl leotard sheer and wet (adult costume)",
            "a devil set with pasties and a tail harness (adult costume)",
        ],
    },
    "club": {
        "cues": ("club", "nightclub", "dance floor", "dancefloor", "rave",
                 "disco", "party", "strobe", "dj", "bass", "neon"),
        "normal": [
            "a sequined party top and high-waisted shorts", "a bodycon mini in a bold print",
            "a metallic bralette and a flip skirt", "a mesh long-sleeve over a solid bra and hot pants",
            "a halter crop and faux-leather shorts", "a chainmail-look top and cigarette pants",
            "a cutout bodysuit and sheer tights", "a fringe mini that whips when she moves",
            "a corset top and a pleated micro skirt", "a rhinestone tube top and vinyl leggings",
            "a one-shoulder crop and a wrap mini", "a holographic two-piece",
            "a slip mini and an open cropped moto jacket", "a backless halter and cargo minis",
            "a ring-linked crop and low-rise flares", "a satin cami and a chain belt over a mini",
            "an asymmetric cutout dress", "a bandeau and shredded-hem denim mini",
            "a glitter mesh tee over a bralette and bike shorts", "a lace-up back top and leather mini",
            "a wet-look catsuit half-zipped", "a rhinestone body chain over a black mini",
            "a sling-strap mini with open sides", "a vinyl bralette and cargo mini",
            "a sheer black long-sleeve over a bikini top and skirt", "a metallic micro dress",
        ],
        "spicy": [
            "rhinestone pasties and a micro skirt", "a sheer mesh dress over a thong",
            "a chain-link bikini top and booty shorts", "a wet-look bra and a barely-there skirt",
            "a cupless bodysuit under an open mesh top", "body glitter and a G-string set",
            "a cage dress with nothing under", "pasties and a rhinestone thong only",
            "oil-slick skin and a micro skirt", "a tearaway club skirt over a G-string",
        ],
    },
    "stage": {
        "cues": ("stage", "concert", "perform", "idol", "showgirl", "cabaret",
                 "burlesque", "pole", "dancer", "spotlight"),
        "normal": [
            "a rhinestone dance two-piece and thigh-high boots", "a fringed flapper mini",
            "a corset leotard and sheer tights", "a feathered showgirl bra and briefs",
            "a sequin bodysuit with a high leg", "an idol stage outfit with a layered skirt",
            "a beaded bralette and a slit skirt", "a satin cabaret corset and stockings",
            "a metallic catsuit unzipped to the sternum", "a crystal-fringe bra and hip belt",
            "a feather-trim mini robe over a sparkle set", "a harness-strap leotard and gloves",
            "a two-piece with detachable tear-away skirt", "a mirrored-disc halter and hot pants",
        ],
        "spicy": [
            "a cupless corset and a tulle skirt", "pasties, a garter, and sheer stockings",
            "a rhinestone G-string and open feather robe", "a chain harness over a sheer bodysuit",
        ],
    },
    "beach": {
        "cues": ("beach", "pool", "swim", "ocean", "sea", "sand", "sunbath",
                 "poolside", "surf", "bikini", "lake"),
        "normal": [
            "a string bikini under an open linen shirt", "a triangle bikini and a sarong",
            "a high-cut one-piece", "a bandeau bikini and cutoff shorts",
            "a crochet cover-up over a dark bikini", "a plunge one-piece with a low back",
            "a wet swimsuit and a towel over one shoulder", "a sporty two-piece and board shorts",
            "an underwire bikini and a straw hat", "a ring-front bikini and an open kimono",
            "a ribbed tank bikini and low-slung linen pants", "a scoop one-piece worn as a top with a wrap skirt",
            "a halter bikini and an oversized shirt sliding off one shoulder", "a strappy multi-string two-piece",
        ],
        "spicy": [
            "a micro bikini gone sheer when wet", "a string bikini untied at the neck",
            "oiled skin and a thong bikini", "a soaked white one-piece turned transparent",
            "just bikini bottoms and crossed arms", "a slipping towel and nothing under it",
        ],
    },
    "gym": {
        "cues": ("gym", "workout", "training", "yoga", "pilates", "fitness",
                 "lifting", "treadmill", "boxing", "sweat", "exercise"),
        "normal": [
            "a scrunch-seam leggings set", "a racerback bra and booty shorts",
            "a one-shoulder sports bra and flared leggings", "a zip-front bra and joggers",
            "a stringer tank over a longline bra", "bike shorts and a cropped windbreaker",
            "a boxing crop and satin trunks", "a strappy-back bra and seamless tights",
            "a matching seamless set in a marl knit", "a crossover-waist flare and twist-front bra",
            "an oversized pump-cover tee over a thong-back onesie", "a cutout back onesie and lifting socks",
            "a ribbed unitard and an open zip hoodie", "a high-neck crop bra and split-hem leggings",
        ],
        "spicy": [
            "a sports bra soaked see-through and low leggings", "a unitard peeled to the waist",
            "just a sports bra and a thong", "damp compression gear clinging everywhere",
        ],
    },
    "bedroom": {
        "cues": ("bed", "bedroom", "sleep", "lingerie", "boudoir", "morning",
                 "sheets", "pillow", "nightstand", "waking"),
        "normal": [
            "a satin robe barely tied", "a camisole and boy-shorts", "a babydoll nightie",
            "silk pyjamas with the top misbuttoned", "a slip nightgown with lace at the hem",
            "a kimono robe over a matching set", "just his shirt, sleeves past her hands",
            "a cropped tee and cotton sleep shorts", "a ribbed henley and bare legs",
            "a satin cami set with feather-trim slippers", "an oversized cardigan over a lace cami",
            "a modal sleep dress clinging soft", "a waffle robe half off one shoulder",
            "a long silk shirt and nothing else visible",
        ],
        "spicy": [
            "a lace bralette and matching thong", "a sheer teddy with nothing under it",
            "a garter belt, stockings, and little else", "an open robe over a G-string",
            "just panties and one of his ties", "a chemise slipping off both shoulders",
        ],
    },
    "office": {
        "cues": ("office", "work", "secretary", "boardroom", "meeting", "desk",
                 "corporate", "boss", "cubicle", "business"),
        "normal": [
            "a pencil skirt and a blouse straining one button", "a fitted blazer dress",
            "a secretary blouse and pleated skirt", "a wrap dress and low heels",
            "a sleeveless shift and a cardigan off the shoulders", "a blouse tucked into wide trousers",
            "a knit polo dress and a slim belt", "a satin shirt half-tucked into a midi skirt",
            "a waistcoat worn as a top with tailored slacks", "a turtleneck bodysuit and a suit skirt",
            "a sheer-sleeve blouse and high-waist trousers", "a shirt dress with the top buttons soft",
        ],
        "spicy": [
            "a business suit with nothing under the blazer", "a blouse open to the waist and a bra",
            "a pencil skirt hiked with stockings showing", "an unbuttoned shirt-dress and lace beneath",
        ],
    },
    "formal": {
        "cues": ("gala", "wedding", "red carpet", "ball", "black tie", "elegant",
                 "cocktail", "evening", "opera"),
        "normal": [
            "a satin cowl-neck gown", "a one-shoulder cocktail dress", "a velvet slip gown",
            "a halter gown with a thigh slit", "a strapless column dress and opera gloves",
            "a backless evening gown", "a beaded flapper-length dress",
            "a corset-bodice ballgown", "a bias-cut silk gown pooling at the floor",
            "a feather-hem cocktail mini", "a draped Grecian gown with a jeweled shoulder",
            "a sheer-overlay gown with a solid slip beneath",
        ],
        "spicy": [
            "a gown slit to the hip with nothing under it", "a sheer-panel dress over bare skin",
            "a plunge gown open to the navel", "a backless gown that skips underwear",
        ],
    },
}


# HIS occasion pools — same detection cues as hers (shared _OCCASIONS keys).
_HIS_OCCASIONS = {
    "club": {
        "normal": [
            "an open silk shirt in a loud print and tailored trousers", "a fitted black tee, chain, and leather trousers",
            "a mesh tank under an open bomber", "a half-buttoned satin shirt and slim slacks",
            "a cropped boxy tee and cargo pants", "a leather vest over bare skin",
            "an unbuttoned Cuban-collar shirt and gold chain", "a fitted turtleneck and pleated trousers",
            "a sleeveless hoodie and joggers", "a sheer black shirt over a dark tank",
        ],
        "spicy": [
            "leather trousers and nothing above the waist", "an open mesh shirt and briefs-line low jeans",
            "a bowtie, cuffs, and bare chest", "unbuttoned jeans and body glitter",
        ],
    },
    "stage": {
        "normal": [
            "a sequin blazer over a bare chest", "a rhinestone-collar shirt open low",
            "a metallic bomber and leather trousers", "a fringe jacket and fitted black jeans",
            "a satin stage shirt with rolled cuffs", "a harness over a fitted tank",
            "a glitter-lapel suit with no shirt", "an open silver shirt and chains",
        ],
        "spicy": [
            "tear-away trousers and a bowtie", "a thong and an open tux jacket (adult stage)",
            "oiled chest, leather shorts, and boots", "a cape and briefs (adult stage)",
        ],
    },
    "beach": {
        "normal": [
            "swim trunks and an open linen shirt", "board shorts and a towel around the neck",
            "a rash guard peeled to the waist", "swim briefs and mirrored sunglasses",
            "wet trunks and salt-dried hair", "an unbuttoned camp shirt and short trunks",
            "a wetsuit unzipped and hanging at the hips", "linen shorts and bare chest",
        ],
        "spicy": [
            "tiny swim briefs riding low", "wet trunks clinging and see-through at the seams",
            "a towel only, knotted low on the hips", "nothing but board shorts undone at the tie",
        ],
    },
    "gym": {
        "normal": [
            "a stringer tank and shorts", "a compression top and joggers",
            "a pump-cover hoodie over a tank", "gym shorts and taped hands",
            "a sweat-dark tee clinging to his back", "a sleeveless hoodie and compression tights",
            "basketball shorts and a backwards cap", "a fitted tank and split shorts",
        ],
        "spicy": [
            "compression shorts and nothing else", "shorts slung low, shirt tucked in the waistband",
            "a jockstrap and gym socks", "sweat-soaked shorts and bare torso",
        ],
    },
    "bedroom": {
        "normal": [
            "grey joggers and nothing else", "a silk robe hanging open",
            "pyjama pants slung low on the hips", "boxers and an open flannel",
            "a waffle henley and sleep shorts", "a towel around the waist, hair damp",
            "an undershirt and loose sleep pants", "a robe half-tied over bare chest",
        ],
        "spicy": [
            "boxer briefs and nothing else", "a sheet at the waist and nothing above",
            "an open robe and briefs", "just unbuttoned pyjama pants",
        ],
    },
    "office": {
        "normal": [
            "a dress shirt with the sleeves rolled and tie loose", "a three-piece with the jacket off",
            "a turtleneck under a blazer", "a crisp shirt and suspenders",
            "an open collar and a waistcoat", "shirtsleeves and a loosened top button",
            "a knit polo and pleated slacks", "a suit with the tie pulled free",
        ],
        "spicy": [
            "a shirt open to the belt and tie undone", "suit trousers, belt open, shirt gone",
            "an unbuttoned waistcoat over bare chest", "a tie and boxers (adult office)",
        ],
    },
    "formal": {
        "normal": [
            "a midnight tux with the bowtie hanging untied", "a velvet dinner jacket",
            "a white-jacket tux and black trousers", "a sharp three-piece with a pocket square",
            "a mandarin-collar tux shirt and slim suit", "a shawl-lapel tux, top button open",
            "a double-breasted suit worn over a bare chest", "a brocade jacket and silk shirt",
        ],
        "spicy": [
            "a tux shirt fully open under the jacket", "bowtie, cummerbund, and nothing else (adult formal)",
            "suit trousers and suspenders over bare skin", "an open tux jacket and briefs (adult formal)",
        ],
    },
}


def detect_occasion(intent: str):
    it = (intent or "").lower()
    for name, spec in _OCCASIONS.items():
        if any(c in it for c in spec["cues"]):
            return name, spec
    return None, None


def _is_explicit(intent: str) -> bool:
    return bool(_EXPLICIT.search(intent or ""))


_HER_ONLY = re.compile(
    r"(?i)\b(bra|bralette|slit|stocking|skirt|dress|gown|heels?|lace|"
    r"neckline|she|her|hers)\b")


def _detail_for(rng, who: str, pool):
    """Pick a lived-in detail the subject could actually be wearing.

    A male seed garment ("a sharp suit with the tie pulled loose") was pairing
    with "the slit finding more leg with each step". The garment pools are
    gendered; this pool never was.
    """
    if (who or "her").strip().lower() != "him":
        return rng.choice(pool)
    safe = [x for x in pool if not _HER_ONLY.search(x)]
    return rng.choice(safe or pool)


def wardrobe_block(who: str, seed: int, intent: str) -> str:
    """Bias paragraph + hard clothes physics, or '' when off / already named."""
    w = (who or "auto").strip().lower()
    if w == "auto":
        w = infer_who(intent)
    if w not in ("her", "him") or not wants_bias(intent):
        return ""
    rng = random.Random(int(seed or 0) ^ 0xC10)
    explicit = _is_explicit(intent)
    occ_name, occ = detect_occasion(intent)

    if w == "her":
        if occ:
            pool = occ["normal"] * 6 + (occ["spicy"] * 6 if explicit else [])
            pool += rng.sample(HER_PIECES, 4)
        else:
            pool = HER_PIECES + (HER_DARING if explicit else [])
        feet_pool = FOOTWEAR_HER
    else:
        hocc = _HIS_OCCASIONS.get(occ_name) if occ_name else None
        if hocc:
            pool = hocc["normal"] * 6 + (hocc["spicy"] * 6 if explicit else [])
            pool += rng.sample(HIS_PIECES, 4)
        else:
            pool = HIS_PIECES + (HIS_DARING if explicit else [])
        feet_pool = FOOTWEAR_HIM

    piece = rng.choice(pool)
    fabric = rng.choice(FABRICS)
    color = rng.choice(COLORS)
    detail = _detail_for(rng, w, DETAILS)
    # ~40% of seeds also pin footwear so LTX doesn't invent random shoes.
    feet = rng.choice(feet_pool) if rng.random() < 0.40 else ""
    subj = "She wears" if w == "her" else "He wears"
    feet_line = (
        f" Footwear lean: {feet} — only if feet or lower legs read on camera; "
        "otherwise leave them out."
        if feet else
        " If feet stay out of frame, do not invent shoes."
    )
    occ_note = (
        f" Occasion lean ({occ_name}): keep the silhouette true to that setting."
        if occ_name else ""
    )
    return (
        "\nWARDROBE (seed bias — invent from this; never write 'from wardrobe' "
        "in the shot)\n"
        f"{subj} something in the territory of {piece}. Seed palette: {color}; "
        f"seed fabric: {fabric} — use either only where it SUITS the garment; "
        f"where it does not, keep the garment and pick the fabric it would "
        f"really be made of. The piece outranks the palette. "
        f"One lived-in detail like {detail}.{feet_line}{occ_note}\n"
        f"{_CLOTHES_PHYSICS}"
    )


# ═══ UNDERWEAR suite ═════════════════════════════════════════════════════════
# When the intent NAMES underwear ("she dances in her panties"), don't suppress
# — reroute: the named garment is law, the seed details only what's unstated
# (style cut + color + fabric + worn detail). Never boring, never overridden.

UNDERWEAR_STYLES = {
    "panties": [
        "a cheeky lace-back cut", "a high-waist brief", "a low-rise string bikini cut",
        "a seamless second-skin cut", "a boyshort", "a side-tie cut",
        "a ruffle-trim cut", "a scalloped-edge hipster", "a sheer-panel brief",
        "a high-leg 90s cut", "a bow-front cut", "a mesh-back cut",
        "a strappy multi-band cut", "a cotton everyday brief worn soft",
        "a satin full brief", "a lace boyleg",
    ],
    "thong": [
        "a whale-tail rising above the waistband", "a micro string", "a lace-front thong",
        "a seamless invisible cut", "a V-string with side rings", "a high-waist strappy thong",
        "a satin tanga", "a chain-side thong",
    ],
    "bra": [
        "a balconette lifting from below", "a plunge with a deep gore", "an unlined sheer cup",
        "a longline bralette", "a strappy cage back", "a demi-cup with lace edging",
        "a front-clasp racerback", "a quarter-cup shelf (adult)", "a triangle soft cup",
        "a bustier-style underwire", "a halter bralette", "a mesh-panel push-up",
    ],
    "lingerie": [
        "a matching balconette-and-cheeky set", "a strappy cage set with gold hardware",
        "a sheer teddy over a thong", "a bustier with garter straps down",
        "a lace bodysuit set", "an unlined mesh set", "a satin cami-and-tap-shorts set",
        "a corseted waist-cincher set", "a bridal-white lace set", "a leather-look harness set",
        "a plunge teddy cut to the navel", "a peekaboo cutout set",
        "a red satin merry-widow", "a black lace babydoll set", "a strappy open-cup set",
        "a pearl-and-lace harness set", "a wet-look vinyl lingerie set",
        "a sheer bodystocking with an open crotch panel", "a schoolgirl plaid bra set (adult)",
        "a nurse white lace set with red cross detail (adult costume)",
        "a bunny-style lingerie set with cuffs (adult costume)",
    ],
    "stockings": [
        "sheer back-seamed stockings", "lace-top hold-ups", "fishnets",
        "sheer black with a garter belt", "white thigh-highs with a bow",
        "glossy sheen tights", "ripped fishnets", "opaque thigh-highs sliding low",
    ],
    "nightwear": [
        "a babydoll skimming the hips", "a satin chemise riding up", "a sheer negligee open over a set",
        "a slip nightie with lace at the bust", "a teddy with snap closures",
        "an oversized-shirt-and-panties combination", "a robe slipping off both shoulders",
        "a lace-trim cami set",
    ],
    "him": [
        "fitted boxer briefs sitting low", "loose boxers slung on the hips",
        "a jockstrap", "trunks with a contrast waistband", "briefs riding the V-line",
        "long-line compression trunks", "silk boxers", "a waistband peeking above open jeans",
    ],
}

_UNDERWEAR_MAP = [
    (re.compile(r"(?i)\b(pant(?:y|ies)|knickers|underwear|undies)\b"), "panties"),
    (re.compile(r"(?i)\b(thong|g-?string|whale\s*tail)\b"), "thong"),
    (re.compile(r"(?i)\b(bra|bralette)\b"), "bra"),
    (re.compile(r"(?i)\b(lingerie|teddy|bodysuit|bustier|garters?|garter\s*belt)\b"), "lingerie"),
    (re.compile(r"(?i)\b(stockings?|fishnets?|thigh[- ]?highs?|hold[- ]?ups?|tights)\b"), "stockings"),
    (re.compile(r"(?i)\b(nightie|negligee|chemise|babydoll|nightgown|nightwear)\b"), "nightwear"),
    (re.compile(r"(?i)\b(boxers?|briefs|jockstrap|trunks)\b"), "him"),
]


def detect_underwear(intent: str):
    """Return (category, matched_word) when the intent names underwear."""
    t = intent or ""
    for rx, cat in _UNDERWEAR_MAP:
        m = rx.search(t)
        if m:
            return cat, m.group(0)
    return None, None


def underwear_block(who: str, seed: int, intent: str) -> str:
    cat, word = detect_underwear(intent)
    if not cat:
        return ""
    rng = random.Random(int(seed or 0) ^ 0x11D)
    style = rng.choice(UNDERWEAR_STYLES[cat])
    fabric = rng.choice(FABRICS)
    color = rng.choice(COLORS)
    detail = _detail_for(rng, cat, DETAILS)
    subj = "his" if cat == "him" else "her"
    return (
        "\nUNDERWEAR (intent's named garment is law — seed fills only gaps)\n"
        f"The {word} the intent names stays as stated. Where silent, lean "
        f"{subj} {word} toward {style} in {color} {fabric}, with one lived-in "
        f"detail like {detail}. Intent-given color/style/state wins.\n"
        "Name how it sits (waistband, cut, cling, bare). Underwear moves with "
        "the body. Keep consistent unless it comes off.\n"
        f"{_CLOTHES_PHYSICS}"
    )


UNDRESS = """
UNDRESS
Clothes come off as a physics chain — never cut-to-nude, never empty beats.
Each removal is its own beat with four named parts: (1) grip — which hand on
which garment, (2) motion — pull/slide/unhook/peel, (3) reveal — what appears,
(4) resting state — floor, ankles, wrist, kicked aside. Outer before under
unless intent says otherwise. What comes off stays off. Fully nude means no
forgotten residuals. Write mid-removal half-on states; do not teleport past them."""
