export type CountryCode = 'TH' | 'US' | 'UK' | 'FR' | 'AU' | 'SG' | 'KR' | 'JP' | 'NO' | 'NZ' | 'IE' | 'TW' | 'MY';

export interface Agent {
  name: string;
  location?: string;
  contact?: string;
  link?: string;
}

export interface BankDetail {
  provider: string;
  accountName: string;
  accountNumber: string;
}

export interface CountryData {
  name: string;
  currency: string;
  price: number;
  shippingFee?: number;
  phonePrefix?: string;
  agents: Agent[];
  bankDetails?: BankDetail[];
  isDirectOrder: boolean;
}

export const COUNTRIES: Record<CountryCode, CountryData> = {
  TH: {
    name: '🇹🇭 Thailand',
    currency: 'THB',
    price: 350,
    shippingFee: 50,
    phonePrefix: '+66',
    isDirectOrder: true,
    agents: [],
    bankDetails: [
      { provider: 'Kasikorn Bank', accountName: 'Ms. Veena Maneesilawong', accountNumber: '1413720792' },
    ],
  },
  US: {
    name: '🇺🇸 United States',
    currency: 'USD',
    price: 15,
    isDirectOrder: false,
    agents: [
      { name: 'Moe Yu Spring Revolution Market', location: 'US West Coast', link: 'https://www.m.me/MYSRevoM' },
      { name: 'Helping Hands for Burma - H2B', location: 'United States', link: 'https://www.m.me/H2BNYC' },
    ],
  },
  UK: {
    name: '🇬🇧 United Kingdom',
    currency: 'GBP',
    price: 15,
    isDirectOrder: false,
    agents: [
      { name: 'Myanmar Accountancy Club UK', link: 'https://www.m.me/MAClubUK' },
    ],
  },
  FR: {
    name: '🇫🇷 France',
    currency: 'EUR',
    price: 15,
    isDirectOrder: false,
    agents: [
      { name: 'Doh Atu - Ensemble pour le Myanmar', link: 'https://www.facebook.com/profile.php?id=100089067352038' },
    ],
  },
  AU: {
    name: '🇦🇺 Australia',
    currency: 'AUD',
    price: 25,
    isDirectOrder: false,
    agents: [
      { name: 'Ma Mon Zin', location: 'Sydney', link: 'https://www.m.me/MonMZin' },
      { name: 'BPLA Support Group - Perth', location: 'Perth', link: 'https://www.facebook.com/profile.php?id=61576540951374' },
    ],
  },
  SG: {
    name: '🇸🇬 Singapore',
    currency: 'SGD',
    price: 20,
    isDirectOrder: false,
    agents: [
      { name: 'ရေချမ်းစင် (SG)', link: 'https://www.facebook.com/profile.php?id=100075701121183' },
    ],
  },
  KR: {
    name: '🇰🇷 Korea',
    currency: 'KRW',
    price: 25000,
    isDirectOrder: false,
    agents: [
      { name: 'Ma Yin Moe Maung', link: 'https://www.m.me/moemaung.yin' },
    ],
  },
  JP: {
    name: '🇯🇵 Japan',
    currency: 'JPY',
    price: 2500,
    isDirectOrder: false,
    agents: [
      { name: 'BPLA Supply Force.JP', link: 'https://www.m.me/bplasupplyforcejp' },
      { name: 'LOVE FOR MYANMAR.JP', link: 'https://www.m.me/loveformyanmarjp' },
    ],
  },
  NO: {
    name: '🇳🇴 Norway',
    currency: 'NOK',
    price: 200,
    isDirectOrder: false,
    agents: [
      { name: 'Myanmar- CRPH Support Group, Norway', link: 'https://www.m.me/crphsupportgroupnorway' },
    ],
  },
  NZ: {
    name: '🇳🇿 New Zealand',
    currency: 'NZD',
    price: 25,
    isDirectOrder: false,
    agents: [
      { name: 'Nway Oo Bazaar', link: 'https://www.m.me/nwayoobazaar' },
    ],
  },
  IE: {
    name: '🇮🇪 Ireland',
    currency: 'EUR',
    price: 15,
    isDirectOrder: false,
    agents: [
      { name: 'CRPH Funding Ireland', link: 'https://www.m.me/crphfundingireland' },
    ],
  },
  TW: {
    name: '🇹🇼 Taiwan',
    currency: 'TWD',
    price: 500,
    isDirectOrder: false,
    agents: [
      { name: '台灣聲援緬甸聯盟 Taiwan Alliance for Myanmar', link: 'https://www.m.me/TaiwanAllianceforMyanmar' },
    ],
  },
  MY: {
    name: '🇲🇾 Malaysia',
    currency: 'MYR',
    price: 50,
    isDirectOrder: false,
    agents: [
      { name: 'Hmine Myo Sat Vocalist', link: 'https://www.m.me/hmine.myo.sat.vocalist' },
    ],
  },
};

export const BOOK_INFO = {
  title: "Beyond The Trenches",
  author: "Maung SaungKha",
  publisher: "Doh Atu Publishing",
  synopsis: "စစ်ဖြစ်နေတဲ့တိုင်းပြည်မှာ ခေတ်ရဲ့ လိုအပ်ချက်အရ ကလောင်ကို ဘေးခဏချပြီး စစ်ထွက်ရတဲ့ ကဗျာဆရာမောင်ဆောင်းခဟာ ယနေ့ခေတ်လူငယ်များစွာရဲ့ ခေတ်ပြိုင်ပုံရိပ်တွေထဲက ကောက်ကြောင်းတခုပါပဲ။ ငြိမ်းချမ်းတဲ့တနေ့ အညာဒေသက ထန်းတောတွေကြားမှာ ကဗျာတွေ ပြန်ရွတ်ဆိုနိုင်မယ့်နေ့ရက်တွေကို မျှော်ရည်ရင်း သူ့ရဲ့ ကဗျာစုစည်းမှု \"ကတုတ်ကျင်းများအလွန်\" မှာ ပါဝင်စီးမျောလိုက်ကြရအောင်...",
  coverImage: "https://lh3.googleusercontent.com/d/1CknfO6HcATeTKOPqruqYTpQ31tMjKbeN",
  stockStatus: "In Stock",
};
