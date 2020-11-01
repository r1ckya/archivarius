import React, {useState} from 'react';
import {
  BrowserRouter as Router,
  Switch,
  Route, Redirect,
} from 'react-router-dom';

import Header from '../Header';

import Footer from "../Footer";
import Upload from "../Upload/Upload";
import View from "../View";
import Search from "../Search";
import UploadResult from "../Upload/UploadResult";
import APIRequest from "../../rest";

const Main = () => {
  const [classes, setClasses] = useState([]);
  if (classes.length === 0) {
    APIRequest('info/classes', []).then(
      function (result) {
        if (result.status === 'ok') {
          let tmp = [];
          delete result.status;
          for (let cls in result) {
            if (result.hasOwnProperty(cls)) {
              tmp.push(result[cls]);
            }
          }
          setClasses(tmp);
        }
      }
    )
  }
  return (
    <div className="Main container-md">
      <Router>
        <Header/>
        <Switch>
          <Route exact path="/">
            {<Redirect to="/search"/>}
          </Route>
          <Route exact path="/upload/">
            <Upload />
          </Route>
          <Route exact path="/upload/:uploadId"  >
          <UploadResult/>
          </Route>
          <Route exact path="/view/:docId" children={<View classes={classes} />}/>
          <Route path="/search">
            <Search classes={classes} />
          </Route>
          <Route exact path="/report">
            {/* todo */}
          </Route>
          <Route path="*">
            <h1>Not Found</h1>
          </Route>
        </Switch>
        <Footer/>
      </Router>
    </div>
  )
};

export default Main;
