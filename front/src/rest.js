import axios from 'axios'

let server = 'http://178.154.225.145:5000'

export default async function APIRequest(method, data, contentType = 'application/json') {
  let response;
  try {
    response = await axios.post(
      server + '/api/' + method,
      data,
      { headers: { 'Content-Type': contentType} }
    );
    console.log(response)
    response.data.status = (response.data && response.data.status) || 'ok';
  } catch (err) {
    console.log(err);
    return {status: 'error'};
  }

  console.log(response.data);
  return response.data;
}
