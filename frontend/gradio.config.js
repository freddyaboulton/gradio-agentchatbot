import prismjs from 'vite-plugin-prismjs';

export default {
  plugins: [
    prismjs({
      languages: 'all',
    })
  ],
  svelte: {
    preprocess: [],
  },
};