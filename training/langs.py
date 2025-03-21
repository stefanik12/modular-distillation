from typing import Optional, List, Tuple

nllb_eng_src_in_tatoeba = ['epo', 'est', 'eus', 'ewe', 'fao', 'fij', 'fin',
                           'fon', 'fra', 'fur', 'gla', 'gle', 'glg', 'grn', 'guj',
                           'hat', 'hau', 'heb', 'hin', 'hne', 'hun', 'hye', 'ibo',
                           'ilo', 'isl', 'ita', 'jav', 'jpn', 'kab', 'kac', 'kam',
                           'kan', 'kas', 'kat', 'kaz', 'kbp', 'kea', 'khm',
                           'kik', 'kin', 'kir', 'kmb', 'kon', 'kor', 'lao', 'lij',
                           'lim', 'lin', 'lit', 'lmo', 'ltz', 'lua', 'lug', 'luo',
                           'lus', 'mag', 'mai', 'mal', 'mar', 'mkd', 'mlt', 'mni',
                           'mos', 'mri', 'mya', 'nld', 'nso', 'nus', 'nya', 'oci',
                           'pag', 'pan', 'pap', 'pol', 'por', 'ron', 'run', 'rus',
                           'sag', 'san', 'sat', 'scn', 'shn', 'sin', 'slk', 'slv',
                           'smo', 'sna', 'snd', 'som', 'sot', 'spa', 'srd', 'ssw',
                           'sun', 'swe', 'szl', 'tam', 'tat', 'tel', 'tgk', 'tgl',
                           'tha', 'tir', 'tpi', 'tsn', 'tso', 'tuk', 'tum', 'tur',
                           'tzm', 'uig', 'ukr', 'umb', 'urd', 'vec', 'vie', 'war',
                           'wol', 'xho', 'yor', 'zho', 'zul']

all_nllb_langs = ['ace', 'ace', 'acm', 'acq', 'aeb', 'afr', 'ajp', 'aka', 'amh', 'apc', 'arb', 'ars', 'ary', 'arz',
                  'asm',
                  'ast', 'awa', 'ayr', 'azb', 'azj', 'bak', 'bam', 'ban', 'bem', 'ben', 'bho', 'bjn', 'bjn', 'bod',
                  'bos',
                  'bug', 'bul', 'cat', 'ceb', 'ces', 'cjk', 'ckb', 'crh', 'cym', 'dan', 'deu', 'dik', 'dyu', 'dzo',
                  'ell',
                  'eng', 'epo', 'est', 'eus', 'ewe', 'fao', 'pes', 'fij', 'fin', 'fon', 'fra', 'fur', 'fuv', 'gla',
                  'gle',
                  'glg', 'grn', 'guj', 'hat', 'hau', 'heb', 'hin', 'hne', 'hrv', 'hun', 'hye', 'ibo', 'ilo', 'ind',
                  'isl',
                  'ita', 'jav', 'jpn', 'kab', 'kac', 'kam', 'kan', 'kas', 'kas', 'kat', 'knc', 'knc', 'kaz', 'kbp',
                  'kea',
                  'khm', 'kik', 'kin', 'kir', 'kmb', 'kon', 'kor', 'kmr', 'lao', 'lvs', 'lij', 'lim', 'lin', 'lit',
                  'lmo',
                  'ltg', 'ltz', 'lua', 'lug', 'luo', 'lus', 'mag', 'mai', 'mal', 'mar', 'min', 'mkd', 'plt', 'mlt',
                  'mni',
                  'khk', 'mos', 'mri', 'zsm', 'mya', 'nld', 'nno', 'nob', 'npi', 'nso', 'nus', 'nya', 'oci', 'gaz',
                  'ory',
                  'pag', 'pan', 'pap', 'pol', 'por', 'prs', 'pbt', 'quy', 'ron', 'run', 'rus', 'sag', 'san', 'sat',
                  'scn',
                  'shn', 'sin', 'slk', 'slv', 'smo', 'sna', 'snd', 'som', 'sot', 'spa', 'als', 'srd', 'srp', 'ssw',
                  'sun',
                  'swe', 'swh', 'szl', 'tam', 'tat', 'tel', 'tgk', 'tgl', 'tha', 'tir', 'taq', 'taq', 'tpi', 'tsn',
                  'tso',
                  'tuk', 'tum', 'tur', 'twi', 'tzm', 'uig', 'ukr', 'umb', 'urd', 'uzn', 'vec', 'vie', 'war', 'wol',
                  'xho',
                  'ydd', 'yor', 'yue', 'zho', 'zho', 'zul']

all_tatoeba_langs = ['eng', 'gbm', 'heb', 'inh', 'kat', 'kum', 'lua', 'mlg', 'nds', 'ofs', 'pms', 'sag', 'snd', 'tet',
                     'tum', 'wln', 'enm', 'gcf', 'her', 'ipk', 'kau', 'kur', 'lug', 'mlt', 'nep', 'oji', 'pmy', 'sah',
                     'som', 'tgk', 'tur', 'wol', 'epo', 'gil', 'hif', 'iro', 'kaz', 'laa', 'luo', 'mni', 'new', 'ood',
                     'pnt', 'san', 'son', 'tgl', 'tvl', 'xal', 'est', 'gla', 'hil', 'isl', 'kbd', 'lad', 'lus', 'mnr',
                     'ngt', 'ori', 'pol', 'sat', 'sot', 'tha', 'tyj', 'xcl', 'eus', 'gle', 'hin', 'ita', 'kbh', 'lah',
                     'lut', 'mnw', 'ngu', 'orm', 'por', 'scn', 'spa', 'tig', 'tyv', 'xho', 'ewe', 'glg', 'hmn', 'ixl',
                     'kbp', 'lao', 'luy', 'moh', 'nhg', 'orv', 'pot', 'sco', 'sqi', 'tir', 'tzl', 'xmf', 'ext', 'glk',
                     'hmo', 'izh', 'kea', 'lat', 'lzz', 'mon', 'nhn', 'osp', 'ppk', 'ses', 'srd', 'tkl', 'tzm', 'yid',
                     'fao', 'glv', 'hne', 'jaa', 'kek', 'lav', 'mad', 'mos', 'nia', 'oss', 'ppl', 'sgn', 'srn', 'tlh',
                     'udm', 'yor', 'fas', 'gor', 'hoc', 'jam', 'kha', 'lbe', 'mag', 'mri', 'niu', 'ota', 'prg', 'sgs',
                     'ssw', 'tly', 'uig', 'yua', 'fij', 'gos', 'hrx', 'jav', 'khm', 'ldn', 'mah', 'msa', 'nld', 'pag',
                     'pus', 'shi', 'stq', 'tmh', 'ukr', 'zap', 'fil', 'got', 'hsb', 'jbo', 'kik', 'lez', 'mai', 'mus',
                     'nlv', 'pai', 'quc', 'shn', 'sun', 'tmr', 'umb', 'zea', 'fin', 'grc', 'hun', 'jdt', 'kin', 'lfn',
                     'mal', 'mvv', 'nnb', 'pal', 'que', 'shs', 'sux', 'toi', 'urd', 'zgh', 'fkv', 'grn', 'hus', 'jiv',
                     'kir', 'lij', 'mam', 'mwl', 'nog', 'pam', 'qya', 'shy', 'swa', 'tok', 'usp', 'zha', 'fon', 'gsw',
                     'hye', 'jpa', 'kjh', 'lim', 'mar', 'mya', 'non', 'pan', 'rap', 'sin', 'swe', 'ton', 'uzb', 'zho',
                     'fra', 'guc', 'hyw', 'jpn', 'kmb', 'lin', 'mdf', 'myv', 'nor', 'pap', 'rhg', 'sjn', 'swg', 'tpi',
                     'vec', 'zul', 'frm', 'guj', 'iba', 'kaa', 'kok', 'lit', 'meh', 'mzn', 'nov', 'pau', 'rif', 'slk',
                     'syl', 'tpw', 'ven', 'zza', 'frp', 'guw', 'ibo', 'kab', 'kom', 'liv', 'mfe', 'nah', 'nqo', 'pck',
                     'roh', 'slv', 'syr', 'trs', 'vie', 'frr', 'hai', 'ido', 'kac', 'kon', 'lkt', 'mgm', 'nap', 'nso',
                     'pcm', 'rom', 'sma', 'szl', 'trv', 'vls', 'fry', 'hat', 'iii', 'kal', 'kor', 'lld', 'mic', 'nau',
                     'nst', 'pdc', 'ron', 'sme', 'tah', 'tsn', 'vol', 'ful', 'hau', 'iku', 'kam', 'krc', 'lmo', 'mik',
                     'nav', 'nus', 'pfl', 'rue', 'sml', 'tam', 'tso', 'vot', 'fur', 'haw', 'ile', 'kan', 'krl', 'lou',
                     'miq', 'nbl', 'nya', 'phn', 'run', 'smn', 'tat', 'tsz', 'wae', 'gag', 'hbo', 'ilo', 'kar', 'ksh',
                     'lrc', 'mix', 'nch', 'oar', 'pih', 'rup', 'smo', 'tcy', 'tts', 'wal', 'gbi', 'hbs', 'ina', 'kas',
                     'kua', 'ltz', 'mkd', 'nde', 'oci', 'pli', 'rus', 'sna', 'tel', 'tuk', 'war']

flores200_langs = [
    "ace_Arab",  "bam_Latn",  "dzo_Tibt",  "hin_Deva",	"khm_Khmr",  "mag_Deva",  "pap_Latn",  "sot_Latn",	"tur_Latn",
    "ace_Latn",  "ban_Latn",  "ell_Grek",  "hne_Deva",	"kik_Latn",  "mai_Deva",  "pbt_Arab",  "spa_Latn",	"twi_Latn",
    "acm_Arab",  "bel_Cyrl",  "eng_Latn",  "hrv_Latn",	"kin_Latn",  "mal_Mlym",  "pes_Arab",  "srd_Latn",	"tzm_Tfng",
    "acq_Arab",  "bem_Latn",  "epo_Latn",  "hun_Latn",	"kir_Cyrl",  "mar_Deva",  "plt_Latn",  "srp_Cyrl",	"uig_Arab",
    "aeb_Arab",  "ben_Beng",  "est_Latn",  "hye_Armn",	"kmb_Latn",  "min_Arab",  "pol_Latn",  "ssw_Latn",	"ukr_Cyrl",
    "afr_Latn",  "bho_Deva",  "eus_Latn",  "ibo_Latn",	"kmr_Latn",  "min_Latn",  "por_Latn",  "sun_Latn",	"umb_Latn",
    "ajp_Arab",  "bjn_Arab",  "ewe_Latn",  "ilo_Latn",	"knc_Arab",  "mkd_Cyrl",  "prs_Arab",  "swe_Latn",	"urd_Arab",
    "aka_Latn",  "bjn_Latn",  "fao_Latn",  "ind_Latn",	"knc_Latn",  "mlt_Latn",  "quy_Latn",  "swh_Latn",	"uzn_Latn",
    "als_Latn",  "bod_Tibt",  "fij_Latn",  "isl_Latn",	"kon_Latn",  "mni_Beng",  "ron_Latn",  "szl_Latn",	"vec_Latn",
    "amh_Ethi",  "bos_Latn",  "fin_Latn",  "ita_Latn",	"kor_Hang",  "mos_Latn",  "run_Latn",  "tam_Taml",	"vie_Latn",
    "apc_Arab",  "bug_Latn",  "fon_Latn",  "jav_Latn",	"lao_Laoo",  "mri_Latn",  "rus_Cyrl",  "taq_Latn",	"war_Latn",
    "arb_Arab",  "bul_Cyrl",  "fra_Latn",  "jpn_Jpan",	"lij_Latn",  "mya_Mymr",  "sag_Latn",  "taq_Tfng",	"wol_Latn",
    "arb_Latn",  "cat_Latn",  "fur_Latn",  "kab_Latn",	"lim_Latn",  "nld_Latn",  "san_Deva",  "tat_Cyrl",	"xho_Latn",
    "ars_Arab",  "ceb_Latn",  "fuv_Latn",  "kac_Latn",	"lin_Latn",  "nno_Latn",  "sat_Olck",  "tel_Telu",	"ydd_Hebr",
    "ary_Arab",  "ces_Latn",  "gaz_Latn",  "kam_Latn",	"lit_Latn",  "nob_Latn",  "scn_Latn",  "tgk_Cyrl",	"yor_Latn",
    "arz_Arab",  "cjk_Latn",  "gla_Latn",  "kan_Knda",	"lmo_Latn",  "npi_Deva",  "shn_Mymr",  "tgl_Latn",	"yue_Hant",
    "asm_Beng",  "ckb_Arab",  "gle_Latn",  "kas_Arab",	"ltg_Latn",  "nso_Latn",  "sin_Sinh",  "tha_Thai",	"zho_Hans",
    "ast_Latn",  "crh_Latn",  "glg_Latn",  "kas_Deva",	"ltz_Latn",  "nus_Latn",  "slk_Latn",  "tir_Ethi",	"zho_Hant",
    "awa_Deva",  "cym_Latn",  "grn_Latn",  "kat_Geor",	"lua_Latn",  "nya_Latn",  "slv_Latn",  "tpi_Latn",	"zsm_Latn",
    "ayr_Latn",  "dan_Latn",  "guj_Gujr",  "kaz_Cyrl",	"lug_Latn",  "oci_Latn",  "smo_Latn",  "tsn_Latn",	"zul_Latn",
    "azb_Arab",  "deu_Latn",  "hat_Latn",  "kbp_Latn",	"luo_Latn",  "ory_Orya",  "sna_Latn",  "tso_Latn",
    "azj_Latn",  "dik_Latn",  "hau_Latn",  "kea_Latn",	"lus_Latn",  "pag_Latn",  "snd_Arab",  "tuk_Latn",
    "bak_Cyrl",  "dyu_Latn",  "heb_Hebr",  "khk_Cyrl",	"lvs_Latn",  "pan_Guru",  "som_Latn",  "tum_Latn"
]

import pycountry


def iso639_3_to_iso639_1(iso639_3_code: str) -> str:
    # Extract the 3-letter code part (before the underscore)
    # Examples:
    # print(iso639_3_to_iso639_1("ces_Latn"))  # Output: cs
    # print(iso639_3_to_iso639_1("eng_Latn"))  # Output: en
    # print(iso639_3_to_iso639_1("zho_Hans"))  # Output: zh

    iso639_3 = iso639_3_code.split('_')[0]

    # Find the corresponding ISO 639-1 code using pycountry
    try:
        lang = pycountry.languages.get(alpha_3=iso639_3)
        if lang and hasattr(lang, 'alpha_2'):
            return lang.alpha_2
        else:
            return f"No ISO 639-1 code for {iso639_3}"
    except LookupError:
        return f"Invalid ISO 639-3 code: {iso639_3}"


def drop_locale(iso639_3_code: str) -> str:
    assert "_" in iso639_3_code, "Given lang does not seem to have a locale assigned"
    return iso639_3_code.split("_")[0]


def get_intersecting_target_langs(tatoeba_target_langs: List[str]) -> Tuple[List[str], List[str]]:
    flores_without_locales = [l.split("_")[0] for l in flores200_langs]
    non_ambig_flores_langs = set([l for l in flores200_langs if flores_without_locales.count(l.split("_")[0]) == 1])
    # print all omitted languages:
    non_ambig_flores_without_locales = set(l.split("_")[0] for l in flores200_langs)

    matching_tatoeba_langs = [l for l in tatoeba_target_langs if l in non_ambig_flores_without_locales]
    covered_flores_langs = [l for l in non_ambig_flores_langs if any(l.startswith(tatoeba_l)
                                                                     for tatoeba_l in matching_tatoeba_langs)]
    # TODO: resolve mismatching number of langs in a full collection (37 vs 38 langs)
    return covered_flores_langs, matching_tatoeba_langs


def match_flores_langs(tatoeba_lang: str) -> List[str]:
    matching_langs = [lang for lang in flores200_langs if lang.startswith(tatoeba_lang)]

    return matching_langs
