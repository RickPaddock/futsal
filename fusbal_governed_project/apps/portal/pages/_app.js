/*
PROV: GREENFIELD.SCAFFOLD.PORTAL.02
REQ: SYS-ARCH-15
WHY: Portal app entrypoint.
*/

import "../styles/portal.css";

export default function App({ Component, pageProps }) {
  return <Component {...pageProps} />;
}

