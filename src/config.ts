export const SITE = {
  website: "https://chefleonwang.github.io", // replace this with your deployed domain
  author: "Leon Wang",
  profile: "https://chefleonwang.github.io",
  desc: "心心念念地中海",
  title: "Chef Leon",
  subtitle_1: "知之 不知",
  subtitle_2: "Stupid Conversation with Smart Chatbot",
  intro_1:"Good good study, day day drink, gym and read:)",
  intro_2:"不观自在，不觉有情",
  intro_3:"不觉有情，不知何处",
  ogImage: "odg.png",
  lightAndDarkMode: true,
  postPerIndex: 999,
  postPerPage: 999,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: true,
    text: "Suggest Changes",
    url: "https://github.com/ChefLeonWang/chefleonwang.github.io/edit/main/",
  },
  dynamicOgImage: true,
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "America/Los_Angeles", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
